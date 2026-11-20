# PRH SSL detector (training + steering demo)

Minimal code for:
- Self-supervised contrastive training of a detector encoder on reconstructed event sets (`prh.train`, `prh.models`, `prh.infonce`)
- Two-point / grid steering utilities (`prh.steer_qwen`) and an interactive demo (`prh.demo_steer`)

## Layout

- `prh/` — package (training, models, data helpers, steering scripts)
- `common.py` — shared helpers (data/cache utilities, PointTransformer used by some backbones)

## Setup

```bash
pip install -r requirements.txt
```

## Train a detector encoder (SSL)

```bash
python -m prh.train --mode detector --seed 0 --steps 3000 --k 128 --out runs/prh_platonic
```

Outputs are written under `--out` (checkpoints/embeddings as implemented in `prh.train`).

## Steering demo

Requires precomputed artifacts (encoder weights/embeddings and cached Qwen hidden states) produced by your pipeline. Then:

```bash
python -m prh.demo_steer
```

See `prh/demo_steer.py` for CLI options and expected paths.

## Notes

- Data paths (caches, Parquet roots) and model paths in the code may need to be edited for your environment.
- This folder is intentionally small: only the training + steering/demo pieces.