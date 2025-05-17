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


## Data Preparation

Download the BYU Motor dataset using the Kaggle CLI and place the files under `data/`.
The directory should look like:

```
<DATA_ROOT>/
  raw/
    train_labels.csv
    train/      # raw training volumes
    test/       # raw test volumes
  processed/
    zarr/
      train/<tomo_id>.zarr
      test/<tomo_id>.zarr
```

Specify the location with the `--data_root` argument (or `BYU_DATA_ROOT` environment variable).

## Training

Run a full training session:

```bash
python -m motor_det.engine.train --data_root data --batch_size 2 --epochs 10
```
`nms_algorithm` controls the NMS method during validation. The default `vectorized` mode automatically switches to `greedy` when detections exceed `--nms_switch_thr`.


Use `--cpu_augment` to perform augmentation on the CPU. When using this flag,
`--pin_memory` can speed up data transfer:

```bash
python -m motor_det.engine.train --data_root data --cpu_augment --pin_memory
```

Training logs and checkpoints are stored under `runs/motor_fold<fold>`.
Monitor progress with:

```bash
tensorboard --logdir runs
```

## Inference

After training, generate predictions with:

```bash
python -m motor_det.engine.infer \
  --weights runs/motor_fold0/best.ckpt \
  --data_root data \
  --out_csv predictions.csv
```

`--batch` and `--num_workers` control throughput. The script automatically
uses the GPU when available.

Quick test scripts such as `quick_train_val.py` and
`motor_det/tests/test_quick_train.py` reproduce small-scale experiments.


Enabling `--pin_memory` is useful when using CPU-based augmentation. When
CUDA augmentation is active (the default), set `--cpu_augment` before enabling
`--pin_memory` to avoid ``RuntimeError: cannot pin 'cuda' memory`` from the
DataLoader.

## Inference

Use the convenience function `run_inference` to load a checkpoint and
predict motor centers for a directory of tomograms.

```python
from motor_det.engine.infer import run_inference, InferConfig

run_inference(
    "weights/best.ckpt",
    data_root="data",
    out_csv="preds.csv",
    cfg=InferConfig(),  # optional configuration overrides
)
```

Alternatively the same defaults are available via the CLI:

```bash
python -m motor_det.engine.infer \
    --weights weights/best.ckpt \
    --data_root data \
    --out_csv preds.csv
```


