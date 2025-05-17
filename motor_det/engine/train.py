"""
Minimal BYU Motor training runner
---------------------------------
Example:
    python -m motor_det.engine.train \
        --data_root data \
        --batch_size 2 \
        --epochs 10 \
        --lr 3e-4
"""
from __future__ import annotations
import argparse
from pathlib import Path

import lightning as L
import torch

from motor_det.data.module import MotorDataModule
from motor_det.engine.lit_module import LitMotorDet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--positive_only", action="store_true")
    p.add_argument("--gpus", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()

    # -------- Data --------
    dm = MotorDataModule(
        data_root=args.data_root,
        fold=args.fold,
        batch_size=args.batch_size,
        num_workers=4,
        positive_only=args.positive_only,
    )
    dm.setup()

    # -------- Model --------
    model = LitMotorDet(
        lr=args.lr,
        weight_decay=args.weight_decay,
        total_steps=len(dm.train_dataloader()) * args.epochs,
    )

    # -------- Trainer --------
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.gpus else "cpu",
        devices=args.gpus if args.gpus else 1,
        precision="16-mixed",          # 사용할 AMP 정밀도
        log_every_n_steps=50,
        default_root_dir=Path("runs") / f"motor_fold{args.fold}",
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
