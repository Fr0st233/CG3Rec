# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict


class ResBlock(nn.Module):
    """
    A Residual Block module (original, unchanged).
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


#  Graph-Aware Label Smoothing (GALS)
def build_cooccurrence_graph(
    all_item_seqs,
    item2id,
    n_items,
    k=10,
    matrix_type='ppmi',
    max_neighbors=None,
    graph_type='undirected',
    window_size=1,
):
    """
    Build item-item graph from user interaction sequences.

    Uses sequential co-occurrence within a configurable forward window to gather counts,
    then converts edge scores by matrix_type:
      - 'ppmi': CSID-style Positive PMI with context smoothing freq^0.75
      - 'cooc': raw co-occurrence count
    Returns neighbor indices and normalized edge weights.

    In directed mode, only forward transitions a→b are counted, and
    neighbors for an item j are its *in-neighbors* (items i where i→j exists).

    Args:
        all_item_seqs (dict): user -> [item1, item2, ...] (raw IDs)
        item2id (dict): raw_item_id -> int_id (1-indexed)
        n_items (int): total number of items (including padding at 0)
        k (int): number of neighbors per item in Top-K mode
        matrix_type (str): 'ppmi' | 'cooc'
        graph_type (str): 'undirected' | 'directed'
        window_size (int): maximum positional distance between co-occurring items

    Returns:
        adj (torch.LongTensor): [n_items, k] - neighbor item indices
        adj_weights (torch.FloatTensor): [n_items, k] - normalized edge weights
    """
    import math

    graph_type = str(graph_type).lower()
    matrix_type = str(matrix_type).lower()
    if matrix_type not in {'ppmi', 'cooc'}:
        raise ValueError(
            f"Unsupported label_smooth_matrix_type={matrix_type}. "
            "Use one of {'ppmi', 'cooc'}."
        )
    if graph_type not in {'undirected', 'directed'}:
        raise ValueError(
            f"Unsupported graph_type={graph_type}. "
            "Use one of {'undirected', 'directed'}."
        )
    window_size = int(window_size)
    if window_size < 1:
        raise ValueError(f'window_size must be >= 1, got {window_size}.')

    cooc = defaultdict(Counter)
    item_freq = Counter()

    for user, items in all_item_seqs.items():
        item_ids = [item2id.get(item, 0) for item in items]

        for item_id in item_ids:
            if item_id > 0:
                item_freq[item_id] += 1

        # Connect forward positions up to window_size apart. Undirected
        # graphs add the reverse edge; directed graphs preserve order.
        for i, a in enumerate(item_ids):
            stop = min(len(item_ids), i + window_size + 1)
            for next_i in range(i + 1, stop):
                b = item_ids[next_i]
                if a > 0 and b > 0 and a != b:
                    cooc[a][b] += 1
                    if graph_type == 'undirected':
                        cooc[b][a] += 1

    total_cooc = sum(sum(v.values())** 0.75 for v in cooc.values())

    if max_neighbors is None:
        max_neighbors = k

    selected_per_item = [[] for _ in range(n_items)]
    width = 0

    # In directed mode, cooc[a][b] means a→b.  For GALS we want *in-neighbors*
    # of the target (items that precede it), so we build an inverted index.
    if graph_type == 'directed':
        predecessors = defaultdict(Counter)
        for a, neighbors in cooc.items():
            for b, cnt in neighbors.items():
                predecessors[b][a] += cnt
        # Replace cooc with predecessors for neighbour lookup
        _lookup = predecessors
    else:
        _lookup = cooc

    for item_id in range(1, n_items):
        if item_id not in _lookup:
            continue

        neighbors_scored = []
        for neighbor, count in _lookup[item_id].items():
            if matrix_type == 'ppmi':
                # Use item_freq from forward transitions for marginal prob.
                # In directed mode, item_freq counts all occurrences in sequences.
                fi = item_freq.get(item_id, 0)
                fj = item_freq.get(neighbor, 0)
                if fi <= 0 or fj <= 0:
                    continue
                score = math.log(
                    (count * total_cooc )
                    / (fi * (fj ** 0.75) + 1e-10)
                    + 1e-10
                )
                score = max(float(score), 0.0)
            else:
                score = float(count)

            if score > 0:
                neighbors_scored.append((neighbor, score))

        neighbors_all = sorted(neighbors_scored, key=lambda x: x[1], reverse=True)
        if not neighbors_all:
            continue

        selected = neighbors_all[:max(1, int(k))]

        selected_per_item[item_id] = selected
        width = max(width, len(selected))

    width = max(1, width)
    adj = torch.zeros(n_items, width, dtype=torch.long)
    adj_weights = torch.zeros(n_items, width, dtype=torch.float32)

    for item_id in range(1, n_items):
        selected = selected_per_item[item_id]
        if not selected:
            continue

        total_score = sum(score for _, score in selected)
        total_score = max(total_score, 1e-12)
        for idx, (neighbor, score) in enumerate(selected):
            adj[item_id, idx] = neighbor
            adj_weights[item_id, idx] = score / total_score

    return adj, adj_weights


class GraphLabelSmoother(nn.Module):
    """
    Graph-aware label smoothing for codebook prediction training.

    For each target item j at codebook level k:
        target = (1-α) * one_hot(c_k^j) + α * mean( one_hot(c_k^n) for n ∈ N(j) )

    where N(j) are j's co-occurrence neighbors and c_k^n is neighbor n's
    code at level k.

    Properties:
    - No extra parameters (only stores precomputed neighbor indices)
    - No extra forward pass
    - Minimal overhead: just scatter_add + weighted CE
    - At α=0: degrades to standard CE (backward compatible)

    Args:
        alpha (float): Smoothing weight. 0 = hard CE, 1 = full smoothing.
        n_neighbors (int): Number of co-occurrence neighbors per item.
        mode (str):
            - 'graph': original GALS neighbor-based smoothing.
            - 'global': distribute alpha uniformly over all non-target classes.
    """
    def __init__(self, alpha=0.1, n_neighbors=10, mode='graph'):
        super().__init__()
        self.alpha = alpha
        self.n_neighbors = n_neighbors
        self.mode = str(mode).lower()
        if self.mode not in {'graph', 'global'}:
            raise ValueError(
                f"Unsupported label smoother mode: {self.mode}. "
                "Use one of {'graph', 'global'}."
            )
        # Global mode does not require precomputed graph buffers.
        self.ready = self.mode == 'global'

    def build(self, adjacency, weights=None, device='cpu'):
        """
        Store co-occurrence adjacency and frequency weights as buffers.

        Args:
            adjacency (torch.LongTensor): [n_items, k] - neighbor item IDs
            weights (torch.FloatTensor or None): [n_items, k] - frequency weights
            device: target device
        """
        K = min(self.n_neighbors, adjacency.shape[1])
        self.register_buffer('neighbor_ids', adjacency[:, :K].to(device))
        if weights is not None:
            self.register_buffer('neighbor_weights', weights[:, :K].to(device))
        else:
            self.register_buffer('neighbor_weights', None)
        self.ready = True

    def smooth_loss(self, logits, target_item_ids, codebook_idx,
                    hard_target, item_id2tokens, codebook_size):
        """
        Compute graph-smoothed cross-entropy loss for one codebook level.

        Args:
            logits: [N, codebook_size] - predicted logits (already temp-scaled)
            target_item_ids: [N] - target item indices (1-indexed int IDs)
            codebook_idx: int - which codebook level
            hard_target: [N] - ground truth code index (0-indexed within codebook)
            item_id2tokens: [n_items, n_codebook] - global token-ID table
            codebook_size: int - vocabulary size per codebook

        Returns:
            loss: scalar - smoothed CE loss
        """
        if self.alpha <= 0:
            return F.cross_entropy(logits, hard_target)

        if self.mode == 'global':
            # Classic global label smoothing: alpha mass uniformly on all
            # non-target classes.
            N, V = logits.shape
            if V <= 1:
                return F.cross_entropy(logits, hard_target)

            target_dist = torch.full(
                (N, V),
                fill_value=self.alpha / (V - 1),
                device=logits.device,
            )
            target_dist.scatter_(
                1,
                hard_target.unsqueeze(1),
                1.0 - self.alpha,
            )

            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(target_dist * log_probs).sum(dim=-1).mean()
            return loss

        if not self.ready:
            return F.cross_entropy(logits, hard_target)

        N, V = logits.shape

        # ---- hard target distribution (one-hot) ----
        hard_dist = torch.zeros(N, V, device=logits.device)
        hard_dist.scatter_(1, hard_target.unsqueeze(1), 1.0)

        # ---- neighbor distribution ----
        neighbor_ids = self.neighbor_ids[target_item_ids]   # [N, K]

        # Get neighbour codes at this codebook level & convert to local index
        neighbor_codes_raw = item_id2tokens[neighbor_ids, codebook_idx]  # [N, K]
        neighbor_local = neighbor_codes_raw - codebook_idx * codebook_size - 1  # [N, K]

        # Mask out padding neighbours (item_id == 0 → invalid)
        valid_mask = (neighbor_ids > 0).float()                     # [N, K]

        # Use frequency weights if available, else uniform
        if self.neighbor_weights is not None:
            w = self.neighbor_weights[target_item_ids] * valid_mask  # [N, K]
            w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-8)
            w = w / w_sum  # re-normalize after masking
        else:
            n_valid = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
            w = valid_mask / n_valid

        soft_dist = torch.zeros(N, V, device=logits.device)
        soft_dist.scatter_add_(
            1,
            neighbor_local.clamp(0, V - 1),
            w
        )
        # soft_dist is already normalized via w

        # ---- smoothed target ----
        target_dist = (1 - self.alpha) * hard_dist + self.alpha * soft_dist

        # ---- soft cross-entropy ----
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(target_dist * log_probs).sum(dim=-1).mean()

        return loss
