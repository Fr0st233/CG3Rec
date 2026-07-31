# CG3Rec

> **Note**: This codebase is built upon [RPG](https://github.com/facebookresearch/RPG_KDD2025) (Hou et al., KDD 2025). We gratefully acknowledge the original authors for their open-source contribution. Our modifications focus on strengthening the memorization capacity of the model by incorporating collaborative signals.

Available categories:
* `Sports_and_Outdoors`
* `Beauty`
* `Toys_and_Games`
* `CDs_and_Vinyl`

Note that:
1. The datasets will be automatically downloaded once the `category` argument is specified.
2. All hyperparameters can be specified via command line arguments. Please refer to:
    * `genrec/default.yaml`
    * `genrec/datasets/AmazonReviews2014/config.yaml`
    * `genrec/models/RPG/config.yaml`

## Reproduction

### Sports and Outdoors

```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=Sports_and_Outdoors \
    --lr=0.003 \
    --temperature=0.03 \
    --n_codebook=16 \
    --num_beams=100 \
    --n_edges=30 \
    --propagation_steps=5 \
    --collab_emb_dim=128 \
    --collab_emb_weight=0.3 \
    --label_smooth_alpha=0.4 \
    --label_smooth_neighbors=100

```

### Beauty

```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=Beauty \
    --lr=0.01 \
    --temperature=0.03 \
    --n_codebook=32 \
    --num_beams=20 \
    --n_edges=200 \
    --propagation_steps=3 \
    --collab_emb_dim=128 \
    --collab_emb_weight=0.3 \
    --label_smooth_alpha=0.4 \
    --label_smooth_neighbors=100
```

### Toys and Games

```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=Toys_and_Games \
    --lr=0.003 \
    --temperature=0.03 \
    --n_codebook=16 \
    --num_beams=200 \
    --n_edges=20 \
    --propagation_steps=3 \
    --collab_emb_dim=128 \
    --collab_emb_weight=0.3 \
    --label_smooth_alpha=0.35 \
    --label_smooth_neighbors=20
```

### CDs and Vinyl

```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=CDs_and_Vinyl \
    --lr=0.001 \
    --temperature=0.03 \
    --n_codebook=64 \
    --num_beams=20 \
    --n_edges=500 \
    --propagation_steps=5 \
    --collab_emb_dim=128 \
    --collab_emb_weight=0.3 \
    --label_smooth_alpha=0.1 \
    --label_smooth_neighbors=100
```



