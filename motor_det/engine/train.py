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
from motor_det.callbacks.freeze_backbone import FreezeBackbone
from motor_det.config import TrainingConfig


def parse_args():
    p = argparse.ArgumentParser(description="BYU Motor training")
    p.add_argument("--config", type=str, help="YAML/JSON configuration file")
    p.add_argument(
        "--env_prefix",
        type=str,
        default="BYU_TRAIN_",
        help="Prefix for environment variable overrides",
    )
    return p.parse_args()


def train(cfg: TrainingConfig):

    # -------- Data --------
    dm = MotorDataModule(
        data_root=cfg.data_root,
        fold=cfg.fold,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.persistent_workers,
        positive_only=cfg.positive_only,
        train_crop_size=cfg.train_crop_size,
        valid_crop_size=cfg.valid_crop_size,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor,
        use_gpu_augment=cfg.use_gpu_augment,
        valid_use_gpu_augment=cfg.valid_use_gpu_augment,
    )
    dm.setup()

    # -------- Model --------
    model = LitMotorDet(
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        total_steps=len(dm.train_dataloader()) * cfg.epochs,
        nms_algorithm=cfg.nms_algorithm,
        nms_switch_thr=cfg.nms_switch_thr,
    )

    if cfg.transfer_weights:
        ckpt = torch.load(cfg.transfer_weights, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)

    # -------- Trainer --------
    callbacks = []
    if cfg.freeze_backbone_epochs > 0:
        callbacks.append(FreezeBackbone(cfg.freeze_backbone_epochs))

    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator="gpu" if cfg.gpus else "cpu",
        devices=cfg.gpus if cfg.gpus else 1,
        precision="16-mixed",          # 사용할 AMP 정밀도
        log_every_n_steps=50,
        default_root_dir=Path("runs") / f"motor_fold{cfg.fold}",
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)


def main() -> None:
    args = parse_args()
    cfg = TrainingConfig.load(args.config, env_prefix=args.env_prefix)
    train(cfg)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
