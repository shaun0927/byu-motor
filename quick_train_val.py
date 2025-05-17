import os
import time
import lightning as L
import torch

from motor_det.engine.train import train
from motor_det.config import TrainingConfig

DATA_ROOT = r"D:\project\Kaggle\BYU\byu-motor\data"
RUNS_DIR = "runs/quick_val"
os.makedirs(RUNS_DIR, exist_ok=True)

L.seed_everything(42)
torch.set_float32_matmul_precision("high")

cfg = TrainingConfig(
    data_root=DATA_ROOT,
    fold=0,
    batch_size=1,
    num_workers=12,
    persistent_workers=True,
    valid_use_gpu_augment=False,
    pin_memory=True,
    epochs=10,
)

start = time.time()
train(cfg)
print(f"\u2714\ufe0e 전체 학습+검증 완료  in {time.time() - start:.1f}s")
