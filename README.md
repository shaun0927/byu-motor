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
python -m motor_det.engine.train --data_root data --batch_size 2 --epochs 10 --lr 3e-4
```
