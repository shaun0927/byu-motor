import os
import time
from pathlib import Path
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import Trainer

from motor_det.data.module import MotorDataModule
from motor_det.engine.lit_module import LitMotorDet

DATA_ROOT = r"D:\project\Kaggle\BYU\byu-motor\data"
RUNS_DIR = "runs/quick_val"
os.makedirs(RUNS_DIR, exist_ok=True)

L.seed_everything(42)
torch.set_float32_matmul_precision("high")

dm = MotorDataModule(
    data_root=DATA_ROOT,
    fold=0,
    batch_size=1,
    num_workers=12,
    persistent_workers=True,
)
dm.setup()

ckpt_cb = ModelCheckpoint(
    dirpath=RUNS_DIR,
    filename="best",
    monitor="val/f2",
    mode="max",
    save_top_k=1,
)

csv_logger = CSVLogger(RUNS_DIR, name="tensorboard")

trainer = Trainer(
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
    max_epochs=10,
    val_check_interval=20,
    limit_val_batches=0.1,
    log_every_n_steps=20,
    callbacks=[RichProgressBar(refresh_rate=20), ckpt_cb],
    logger=csv_logger,
)

start = time.time()
trainer.fit(LitMotorDet(), dm)
print(f"\u2714\ufe0e 전체 학습+검증 완료  in {time.time() - start:.1f}s")
print("Best checkpoint:", ckpt_cb.best_model_path)
