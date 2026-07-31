# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import json
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import svds
from collections import defaultdict, Counter
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class RPGTokenizer(AbstractTokenizer):
    """
    An example when "codebook_size == 256, n_codebooks == 32":
        0: padding
        1-256: digit 1
        257-512: digit 2
        ...
        7937-8192: digit 32
        8193: eos

    Args:
        config (dict): The configuration dictionary.
        dataset (AbstractDataset): The dataset object.

    Attributes:
        n_codebook_bits (int): The number of bits for the codebook.
        index_factory (str): The index factory name for the OPQ algorithm.
        item2tokens (dict): A dictionary mapping items to their semantic IDs.
        base_user_id (int): The base user ID.
        n_user_tokens (int): The number of user tokens.
        eos_token (int): The end-of-sequence token.
    """
    def __init__(self, config: dict, dataset: AbstractDataset):
        self.n_codebook_bits = self._get_codebook_bits(config['codebook_size'])
        self.index_factory = f'OPQ{config["n_codebook"]},IVF1,PQ{config["n_codebook"]}x{self.n_codebook_bits}'

        super(RPGTokenizer, self).__init__(config, dataset)
        self.item2id = dataset.item2id
        self.user2id = dataset.user2id
        self.id2item = dataset.id_mapping['id2item']
        self.item2tokens = self._init_tokenizer(dataset)
        self.eos_token = self.n_digit * self.codebook_size + 1
        self.ignored_label = -100

    @property
    def n_digit(self):
        """
        Returns the number of digits for the tokenizer.

        The number of digits is determined by the value of `rq_n_codebooks` in the configuration.
        """
        return self.config['n_codebook']

    @property
    def codebook_size(self):
        """
        Returns an integer representing the number of codebooks for the tokenizer.
        """
        return self.config['codebook_size']

    @property
    def max_token_seq_len(self) -> int:
        """
        Returns the maximum token sequence length, including the EOS token.

        Returns:
            int: The maximum token sequence length.
        """
        return self.config['max_item_seq_len']

    @property
    def vocab_size(self) -> int:
        """
        Returns the vocabulary size for the TIGER tokenizer.
        """
        return self.eos_token + 1

    def _get_codebook_bits(self, n_codebook):
        x = math.log2(n_codebook)
        assert x.is_integer() and x >= 0, "Invalid value for n_codebook"
        return int(x)

    def _encode_sent_emb(self, dataset: AbstractDataset, output_path: str):
        """
        Encodes the sentence embeddings for the given dataset and saves them to the specified output path.

        Args:
            dataset (AbstractDataset): The dataset containing the sentences to encode.
            output_path (str): The path to save the encoded sentence embeddings.

        Returns:
            numpy.ndarray: The encoded sentence embeddings.
        """
        assert self.config['metadata'] == 'sentence', \
            'TIGERTokenizer only supports sentence metadata.'

        meta_sentences = [] # 1-base, meta_sentences[0] -> item_id = 1
        for i in range(1, dataset.n_items):
            meta_sentences.append(dataset.item2meta[dataset.id_mapping['id2item'][i]])

        if 'sentence-transformers' in self.config['sent_emb_model']:
            sent_emb_model = SentenceTransformer(
                self.config['sent_emb_model']
            ).to(self.config['device'])

            sent_embs = sent_emb_model.encode(
                meta_sentences,
                convert_to_numpy=True,
                batch_size=self.config['sent_emb_batch_size'],
                show_progress_bar=True,
                device=self.config['device']
            )
        elif 'text-embedding-3' in self.config['sent_emb_model']:
            from openai import OpenAI
            client = OpenAI(api_key=self.config['openai_api_key'])

            sent_embs = []
            for i in tqdm(range(0, len(meta_sentences), self.config['sent_emb_batch_size']), desc='Encoding'):
                try:
                    responses = client.embeddings.create(
                        input=meta_sentences[i: i + self.config['sent_emb_batch_size']],
                        model=self.config['sent_emb_model']
                    )
                except:
                    self.log(f'[TOKENIZER] Failed to encode sentence embeddings for {i} - {i + self.config["sent_emb_batch_size"]}')
                    batch = meta_sentences[i: i + self.config['sent_emb_batch_size']]

                    from genrec.utils import num_tokens_from_string
                    new_batch = []
                    for sent in batch:
                        n_tokens = num_tokens_from_string(sent, 'cl100k_base')
                        if n_tokens < 8192:
                            new_batch.append(sent)
                        else:
                            n_chars = 8192 / n_tokens * len(sent) - 100
                            new_batch.append(sent[:int(n_chars)])

                    self.log(f'[TOKENIZER] Retrying with {len(new_batch)} sentences')
                    responses = client.embeddings.create(
                        input=new_batch,
                        model=self.config['sent_emb_model']
                    )

                for response in responses.data:
                    sent_embs.append(response.embedding)
            sent_embs = np.array(sent_embs, dtype=np.float32)

        sent_embs.tofile(output_path)
        return sent_embs

    def _load_pretrained_collaborative_embedding(self, dataset, output_path):
        """Load externally provided item embeddings for the GET fusion branch.

        Supported formats:
          - .npy: dense array of shape [n_items-1, dim] or [n_items, dim] with padding row 0
          - .npz: contains `embeddings` (and optionally `item_ids`)
          - .pt / .pth: tensor or dict with `embeddings`

        If item_ids are provided, they are reordered to dataset id order.
        Otherwise we assume the rows are already aligned to item IDs.
        """
        import os
        import numpy as _np
        import torch as _torch

        emb_path = self.config.get('pretrained_item_emb_path', '')
        if not emb_path:
            raise ValueError(
                'collab_emb_source=pretrained but pretrained_item_emb_path is empty.'
            )
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f'pretrained_item_emb_path not found: {emb_path}')

        ext = os.path.splitext(emb_path)[1].lower()
        item_ids = None
        if ext == '.npy':
            embs = _np.load(emb_path)
        elif ext == '.npz':
            data = _np.load(emb_path, allow_pickle=True)
            embs = data['embeddings']
            if 'item_ids' in data:
                item_ids = data['item_ids']
        elif ext in {'.pt', '.pth'}:
            obj = _torch.load(emb_path, map_location='cpu')
            if isinstance(obj, dict):
                if 'embeddings' in obj:
                    embs = obj['embeddings']
                elif 'item_embs' in obj:
                    embs = obj['item_embs']
                else:
                    raise KeyError(
                        "Torch checkpoint must contain 'embeddings' or 'item_embs'."
                    )
                item_ids = obj.get('item_ids', None)
            else:
                embs = obj
        else:
            raise ValueError(
                f'Unsupported pretrained_item_emb_path extension: {ext}. '
                'Use .npy, .npz, .pt, or .pth.'
            )

        if isinstance(embs, _torch.Tensor):
            embs = embs.detach().cpu().numpy()
        embs = _np.asarray(embs, dtype=_np.float32)

        if item_ids is not None:
            if isinstance(item_ids, _torch.Tensor):
                item_ids = item_ids.detach().cpu().numpy()
            item_ids = _np.asarray(item_ids)
            aligned = _np.zeros((dataset.n_items - 1, embs.shape[1]), dtype=_np.float32)
            if len(item_ids) != len(embs):
                raise ValueError('item_ids and embeddings length mismatch in pretrained_item_emb_path.')
            for idx, item_id in enumerate(item_ids):
                item_id = int(item_id)
                if item_id <= 0 or item_id >= dataset.n_items:
                    continue
                aligned[item_id - 1] = embs[idx]
            embs = aligned
        else:
            if embs.shape[0] == dataset.n_items:
                embs = embs[1:]
            elif embs.shape[0] != dataset.n_items - 1:
                raise ValueError(
                    f'Pretrained embeddings row count mismatch: got {embs.shape[0]}, '
                    f'expected {dataset.n_items - 1} or {dataset.n_items}.'
                )

        norms = _np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / _np.maximum(norms, 1e-8)
        embs.tofile(output_path)
        self.log(
            f'[GET] Loaded pretrained item embeddings from {emb_path}, '
            f'shape={embs.shape}'
        )
        return embs


    def _build_collaborative_embedding(self, dataset, dim, output_path, matrix_type='ppmi'):
        """
        Build collaborative embeddings via matrix + SVD on co-occurrence data.

        Uses TRAINING interactions only (strip last 2 items per user to avoid
        leaking val/test targets — matching the GALS graph construction).

        Pipeline:
            1. Build item co-occurrence matrix from sequential interactions
            2. Build sparse matrix by selected type:
               - 'ppmi': Positive PMI transformed matrix
               - 'cooc': raw co-occurrence count matrix
            3. Truncated SVD to get dense embeddings
            4. L2-normalize

        Args:
            dataset: The dataset object with all_item_seqs.
            dim: Embedding dimension for collaborative vectors.
            output_path: Path to cache the resulting embeddings.

        Returns:
            np.ndarray of shape [n_items-1, dim], float32, L2-normalized.
        """
        n_items = dataset.n_items  # includes padding item 0
        matrix_type = str(matrix_type).lower()
        if matrix_type not in {'ppmi', 'cooc'}:
            raise ValueError(
                f"Unsupported collab_matrix_type={matrix_type}. "
                "Use one of {'ppmi', 'cooc'}."
            )

        # Step 1: Build co-occurrence count matrix from training sequences
        graph_type = str(self.config.get('graph_type', 'undirected')).lower()
        window_size = int(self.config.get('cooc_window_size', 1))
        if window_size < 1:
            raise ValueError(
                f'cooc_window_size must be >= 1, got {window_size}.'
            )
        self.log(
            f'[GET] Building co-occurrence matrix '
            f'(graph_type={graph_type}, window_size={window_size}) '
            f'from training data...'
        )
        cooc = defaultdict(Counter)
        item_freq = Counter()

        for user, items in dataset.all_item_seqs.items():
            train_items = items[:-2] if len(items) > 2 else []
            # Convert to IDs
            id_seq = [dataset.item2id[it] for it in train_items if it in dataset.item2id]
            # Count frequencies
            for iid in id_seq:
                item_freq[iid] += 1
            for j, a in enumerate(id_seq):
                stop = min(len(id_seq), j + window_size + 1)
                for next_j in range(j + 1, stop):
                    b = id_seq[next_j]
                    if a != b:  # skip self-loops
                        cooc[a][b] += 1
                        if graph_type == 'undirected':
                            cooc[b][a] += 1

        # Step 2: Build sparse matrix (PPMI or raw co-occurrence)
        if matrix_type == 'ppmi':
            self.log(f'[GET] Computing PPMI matrix...')
        else:
            self.log(f'[GET] Building raw co-occurrence matrix...')
        total_cooc = sum(sum(v.values())** 0.75 for v in cooc.values())
        if total_cooc == 0:
            self.log('[GET] WARNING: No co-occurrences found, returning zero embeddings')
            embs = np.zeros((n_items - 1, dim), dtype=np.float32)
            embs.tofile(output_path)
            return embs

        rows, cols, data = [], [], []
        for i, neighbors in cooc.items():
            for j, count in neighbors.items():
                if matrix_type == 'ppmi':
                    pmi = np.log(
                        (count * total_cooc)
                        / (item_freq[i] * (item_freq[j] ** 0.75) + 1e-10)
                        + 1e-10
                    )
                    value = max(pmi, 0.0)  # Positive PMI
                    if value <= 0:
                        continue
                else:
                    # Raw co-occurrence count (ablation baseline).
                    value = float(count)
                    if value <= 0:
                        continue

                rows.append(i - 1)  # item IDs are 1-based, matrix is 0-based
                cols.append(j - 1)
                data.append(value)

        collab_matrix = sp.csr_matrix(
            (data, (rows, cols)), shape=(n_items - 1, n_items - 1)
        )
        self.log(
            f'[GET] {matrix_type.upper()} matrix: {collab_matrix.nnz} non-zeros, '
            f'density={collab_matrix.nnz / (n_items - 1)**2:.6f}'
        )

        actual_dim = min(dim, min(collab_matrix.shape) - 2)
        self.log(f'[GET] Running SVD with dim={actual_dim}...')
        U, S, Vt = svds(collab_matrix.astype(np.float64), k=actual_dim)

        embs = U * np.sqrt(S)[np.newaxis, :]  # [n_items-1, dim]
        embs = embs.astype(np.float32)

        # L2-normalize
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embs = embs / norms

        # Pad to requested dim if needed (SVD gave fewer dims)
        if embs.shape[1] < dim:
            pad = np.zeros((embs.shape[0], dim - embs.shape[1]), dtype=np.float32)
            embs = np.concatenate([embs, pad], axis=1)

        self.log(f'[GET] Collaborative embeddings shape: {embs.shape}, '
                 f'mean={embs.mean():.4f}, std={embs.std():.4f}')

        embs.tofile(output_path)
        return embs

    def _get_items_for_training(self, dataset: AbstractDataset) -> np.ndarray:
        """
        Get a boolean mask indicating which items are used for training.

        Args:
            dataset (AbstractDataset): The dataset containing the item sequences.

        Returns:
            np.ndarray: A boolean mask indicating which items are used for training.
        """
        items_for_training = set()
        for item_seq in dataset.split_data['train']['item_seq']:
            for item in item_seq:
                items_for_training.add(item)
        self.log(f'[TOKENIZER] Items for training: {len(items_for_training)} of {dataset.n_items - 1}')
        mask = np.zeros(dataset.n_items - 1, dtype=bool)
        for item in items_for_training:
            mask[dataset.item2id[item] - 1] = True
        return mask

    def _generate_semantic_id_opq(self, sent_embs, sem_ids_path, train_mask):
        """
        Generates semantic IDs using the OPQ algorithm.

        Args:
            sent_embs (numpy.ndarray): Array of sentence embeddings.
            sem_ids_path (str): Path to save the generated semantic IDs.
            train_mask (numpy.ndarray): Boolean mask indicating the training samples.
        """
        import faiss
        if self.config['opq_use_gpu']:
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024 * 1024 * 512)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = self.n_digit >= 56
        faiss.omp_set_num_threads(self.config['faiss_omp_num_threads'])
        index = faiss.index_factory(
            sent_embs.shape[1],
            self.index_factory,
            faiss.METRIC_INNER_PRODUCT
        )
        self.log(f'[TOKENIZER] Training index...')
        if self.config['opq_use_gpu']:
            index = faiss.index_cpu_to_gpu(res, self.config['opq_gpu_id'], index, co)
        index.train(sent_embs[train_mask])
        index.add(sent_embs)
        if self.config['opq_use_gpu']:
            index = faiss.index_gpu_to_cpu(index)

        ivf_index = faiss.downcast_index(index.index)
        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
        pq_codes = pq_codes.reshape(-1, invlists.code_size)

        faiss_sem_ids = []
        n_bytes = pq_codes.shape[1]
        for u8code in pq_codes:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
            code = []
            for i in range(self.n_digit):
                code.append(bs.read(self.n_codebook_bits))
            faiss_sem_ids.append(code)
        pq_codes = np.array(faiss_sem_ids)

        item2sem_ids = {}
        for i in range(pq_codes.shape[0]):
            item = self.id2item[i + 1]
            item2sem_ids[item] = tuple(pq_codes[i].tolist())
        self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)

    def _sem_ids_to_tokens(self, item2sem_ids: dict) -> dict:
        """
        Converts semantic IDs to tokens.

        Args:
            item2sem_ids (dict): A dictionary mapping items to their corresponding semantic IDs.

        Returns:
            dict: A dictionary mapping items to their corresponding tokens.
        """
        for item in item2sem_ids:
            tokens = list(item2sem_ids[item])
            for digit in range(self.n_digit):
                # "+ 1" as 0 is reserved for padding
                tokens[digit] += self.codebook_size * digit + 1
            item2sem_ids[item] = tuple(tokens)
        return item2sem_ids

    def _init_tokenizer(self, dataset: AbstractDataset):
        """
        Initialize the tokenizer.

        Args:
            dataset (AbstractDataset): The dataset object.

        Returns:
            dict: A dictionary mapping items to semantic IDs.
        """
        # Load semantic IDs (path includes GET params when enabled)
        collab_dim = self.config.get('collab_emb_dim', 0)
        collab_weight = self.config.get('collab_emb_weight', 0)
        collab_matrix_type = str(self.config.get('collab_matrix_type', 'ppmi')).lower()
        collab_noise = self.config.get('collab_emb_noise', False)
        cooc_window_size = int(self.config.get('cooc_window_size', 1))
        if cooc_window_size < 1:
            raise ValueError(
                f'cooc_window_size must be >= 1, got {cooc_window_size}.'
            )
        matrix_tag = f'_m{collab_matrix_type}' if collab_matrix_type != 'ppmi' else ''
        noise_tag = '_noise' if collab_noise else ''
        pretrain_tag = '_pretrain' if self.config.get('collab_emb_source', 'svd') == 'pretrained' else ''
        graph_tag = '_dir' if str(self.config.get('graph_type', 'undirected')).lower() == 'directed' else ''
        window_tag = f'_win{cooc_window_size}' if cooc_window_size != 1 else ''
        GET_suffix = (f'_GET{collab_dim}w{collab_weight}{matrix_tag}{noise_tag}{pretrain_tag}{graph_tag}{window_tag}'
                       if collab_dim > 0 else '')
        sem_ids_path = os.path.join(
            dataset.cache_dir, 'processed',
            f'{os.path.basename(self.config["sent_emb_model"])}_{self.index_factory}{GET_suffix}.sem_ids'
        )

        if not os.path.exists(sem_ids_path):
            # Load or encode sentence embeddings
            sent_emb_path = os.path.join(
                dataset.cache_dir, 'processed',
                f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb'
            )
            if os.path.exists(sent_emb_path):
                self.log(f'[TOKENIZER] Loading sentence embeddings from {sent_emb_path}...')
                sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(-1, self.config['sent_emb_dim'])
            else:
                self.log(f'[TOKENIZER] Encoding sentence embeddings...')
                sent_embs = self._encode_sent_emb(dataset, sent_emb_path)
            # PCA
            if self.config['sent_emb_pca'] > 0:
                self.log(f'[TOKENIZER] Applying PCA to sentence embeddings...')
                from sklearn.decomposition import PCA
                pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
                sent_embs = pca.fit_transform(sent_embs)
            self.log(f'[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}')

            # ===== GET: Fuse collaborative embeddings =====
            collab_dim = self.config.get('collab_emb_dim', 0)
            collab_weight = self.config.get('collab_emb_weight', 0.3)
            if collab_dim > 0:
                collab_source = str(self.config.get('collab_emb_source', 'svd')).lower()
                collab_noise = self.config.get('collab_emb_noise', False)
                if collab_noise:
                    # === Noise Control: random Gaussian instead of real collab ===
                    n_items_emb = sent_embs.shape[0]
                    rng = np.random.RandomState(42)  # fixed seed for reproducibility
                    collab_embs = rng.randn(n_items_emb, collab_dim).astype(np.float32)
                    noise_norms = np.linalg.norm(collab_embs, axis=1, keepdims=True)
                    collab_embs = collab_embs / np.maximum(noise_norms, 1e-8)
                    self.log(f'[GET-NOISE] Using RANDOM noise ({collab_dim}d) '
                             f'instead of collaborative embeddings (control experiment)')
                else:
                    if collab_source == 'pretrained':
                        collab_emb_path = os.path.join(
                            dataset.cache_dir, 'processed',
                            f'collab_pretrained_d{collab_dim}.emb'
                        )
                        if os.path.exists(collab_emb_path):
                            self.log(f'[GET] Loading pretrained collaborative embeddings from {collab_emb_path}...')
                            collab_embs = np.fromfile(collab_emb_path, dtype=np.float32).reshape(-1, collab_dim)
                        else:
                            collab_embs = self._load_pretrained_collaborative_embedding(
                                dataset,
                                collab_emb_path,
                            )
                    else:
                        collab_emb_path = os.path.join(
                            dataset.cache_dir, 'processed',
                            f'collab_{collab_matrix_type}_svd_d{collab_dim}{graph_tag}{window_tag}.emb'
                        )
                        if os.path.exists(collab_emb_path):
                            self.log(f'[GET] Loading collaborative embeddings from {collab_emb_path}...')
                            collab_embs = np.fromfile(collab_emb_path, dtype=np.float32).reshape(-1, collab_dim)
                        else:
                            collab_embs = self._build_collaborative_embedding(
                                dataset,
                                collab_dim,
                                collab_emb_path,
                                matrix_type=collab_matrix_type,
                            )


                # L2-normalize semantic embeddings before fusion
                sem_norms = np.linalg.norm(sent_embs, axis=1, keepdims=True)
                sent_embs_normed = sent_embs / np.maximum(sem_norms, 1e-8)

                # Weighted concatenation: [sem * (1-w), collab * w]
                fused = np.concatenate([
                    sent_embs_normed * (1.0 - collab_weight),
                    collab_embs * collab_weight,
                ], axis=1)
                self.log(
                    f'[GET] Fused embeddings: semantic({sent_embs.shape[1]}d) + '
                    f'collab({collab_dim}d) → {fused.shape[1]}d, '
                    f'weight={collab_weight}'
                )
                sent_embs = fused

            # Generate semantic IDs
            training_item_mask = self._get_items_for_training(dataset)
            self._generate_semantic_id_opq(sent_embs, sem_ids_path, training_item_mask)

        self.log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
        item2sem_ids = json.load(open(sem_ids_path, 'r'))
        item2tokens = self._sem_ids_to_tokens(item2sem_ids)

        return item2tokens

    def _tokenize_first_n_items(self, item_seq: list) -> tuple:
        """
        Tokenizes the first n items in the given item_seq.
        The losses for the first n items can be computed by only forwarding once.

        Args:
            item_seq (list): The item sequence that contains the first n items.

        Returns:
            tuple: A tuple containing the tokenized input_ids, attention_mask, labels, and seq_lens.
        """
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens

        pad_lens = self.max_token_seq_len - seq_lens
        input_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)

        labels = [self.item2id[item] for item in item_seq[1:]]
        labels.extend([self.ignored_label] * pad_lens)

        return input_ids, attention_mask, labels, seq_lens

    def _tokenize_later_items(self, item_seq: list, pad_labels: bool = True) -> tuple:
        """
        Tokenizes the later items in the item sequence.
        Only the last one items are used as the target item.

        Args:
            item_seq (list): The item sequence.

        Returns:
            tuple: A tuple containing the tokenized input IDs, attention mask, labels, and seq_lens.
        """
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens
        labels = [self.ignored_label] * seq_lens
        labels[-1] = self.item2id[item_seq[-1]]

        pad_lens = self.max_token_seq_len - seq_lens
        input_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)
        if pad_labels:
            labels.extend([self.ignored_label] * pad_lens)

        return input_ids, attention_mask, labels, seq_lens

    def tokenize_function(self, example: dict, split: str) -> dict:
        """
        Tokenizes the input example based on the specified split.

        Args:
            example (dict): The input example containing the item sequence.
            split (str): The split type ('train' or 'val' or 'test').

        Returns:
            dict: A dictionary containing the tokenized input, attention mask, and labels.
        """
        max_item_seq_len = self.config['max_item_seq_len']
        item_seq = example['item_seq'][0]
        if split == 'train':
            n_return_examples = max(len(item_seq) - max_item_seq_len, 1)

            # Tokenize the first n items if len(item_seq) <= max_item_seq_len + 1
            input_ids, attention_mask, labels, seq_lens = self._tokenize_first_n_items(
                # Add 1 as the target item is not included in the input sequence
                item_seq=item_seq[:min(len(item_seq), max_item_seq_len + 1)]
            )
            all_input_ids, all_attention_mask, all_labels, all_seq_lens = \
                [input_ids], [attention_mask], [labels], [seq_lens]

            # Tokenize the later items if len(item_seq) > max_item_seq_len + 1
            for i in range(1, n_return_examples):
                cur_item_seq = item_seq[i:i+max_item_seq_len+1]
                input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(cur_item_seq)
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
                all_seq_lens.append(seq_lens)

            return {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_mask,
                'labels': all_labels,
                'seq_lens': all_seq_lens,
            }
        else:
            input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(
                item_seq=item_seq[-(max_item_seq_len+1):],
                pad_labels=False
            )
            return {
                'input_ids': [input_ids],
                'attention_mask': [attention_mask],
                'labels': [labels[-1:]],
                'seq_lens': [seq_lens]
            }

    def tokenize(self, datasets: dict) -> dict:
        """
        Tokenizes the given datasets.

        Args:
            datasets (dict): A dictionary of datasets to tokenize.

        Returns:
            dict: A dictionary of tokenized datasets.
        """
        tokenized_datasets = {}
        for split in datasets:
            tokenized_datasets[split] = datasets[split].map(
                lambda t: self.tokenize_function(t, split),
                batched=True,
                batch_size=1,
                remove_columns=datasets[split].column_names,
                num_proc=self.config['num_proc'],
                desc=f'Tokenizing {split} set: '
            )

        for split in datasets:
            tokenized_datasets[split].set_format(type='torch')

        return tokenized_datasets
