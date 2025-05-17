import os
import time
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from motor_det.data.module import MotorDataModule
from motor_det.engine.lit_module import LitMotorDet

# Quick runtime test inspired by quick_train.ipynb
DATA_ROOT = os.environ.get("BYU_DATA_ROOT", "data")
RUNS_DIR = "runs/quick_test"
FOLD = 0
os.makedirs(RUNS_DIR, exist_ok=True)

L.seed_everything(42)
torch.set_float32_matmul_precision("high")

# Only a small fraction of data is used
dm = MotorDataModule(
    data_root=DATA_ROOT,
    fold=FOLD,
    batch_size=1,
    num_workers=12,
    persistent_workers=True,
)
dm.setup()

model = LitMotorDet(lr=2e-4, total_steps=1_000)

ckpt_cb = ModelCheckpoint(
    dirpath=RUNS_DIR,
    filename="best",
    monitor="val/f2",
    mode="max",
    save_top_k=1,
)

trainer = L.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    precision="16-mixed",
    max_epochs=1,
    limit_train_batches=0.01,
    limit_val_batches=0.01,
    check_val_every_n_epoch=1,
    callbacks=[RichProgressBar(), ckpt_cb],
    log_every_n_steps=20,
)

start = time.time()
trainer.fit(model, dm)
print(f"\u2714\ufe0e training done in {time.time()-start:0.1f}s")
print("saved:", ckpt_cb.best_model_path)

