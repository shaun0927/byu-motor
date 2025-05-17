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

Enabling `--pin_memory` is useful when using CPU-based augmentation. When
CUDA augmentation is active (the default), set `--cpu_augment` before enabling
`--pin_memory` to avoid ``RuntimeError: cannot pin 'cuda' memory`` from the
DataLoader.
