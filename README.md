# BYU Motor Detection

BYU motor detection project for detecting defects in BYU dataset.

## Setup

Install with conda using `environment.yml`:

```bash
conda env create -f environment.yml
conda activate byu-motor
```

Alternatively install via `pyproject.toml`:

```bash
pip install -e .
```

## Training

Run the training script:

```bash
python -m motor_det.engine.train --data_root data --batch_size 2 --epochs 10 \
    --lr 3e-4 [--positive_only] \
    [--nms_algorithm vectorized] [--nms_switch_thr 1000]
```

``nms_algorithm`` controls the NMS method used during validation. The
default ``vectorized`` algorithm automatically falls back to ``greedy`` when
more than ``nms_switch_thr`` proposals are generated (default 1,000).

Enabling `--pin_memory` is useful when using CPU-based augmentation. When
CUDA augmentation is active (the default), set `--cpu_augment` before enabling
`--pin_memory` to avoid ``RuntimeError: cannot pin 'cuda' memory`` from the
DataLoader.
