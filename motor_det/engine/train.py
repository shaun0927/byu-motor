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
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Training config file")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--positive_only", action="store_true")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--train_depth_window_size", type=int, default=96)
    p.add_argument("--train_spatial_window_size", type=int, default=128)
    p.add_argument("--valid_depth_window_size", type=int, default=192)
    p.add_argument("--valid_spatial_window_size", type=int, default=128)
    p.add_argument("--transfer_weights", type=str, default=None)
    p.add_argument("--freeze_backbone_epochs", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true", help="Enable DataLoader pin_memory")
    p.add_argument("--prefetch_factor", type=int, default=None)
    p.add_argument("--cpu_augment", action="store_true", help="Run augmentation on CPU")
    p.add_argument("--mixup", type=float, default=0.0, help="MixUp probability")
    p.add_argument("--cutmix", type=float, default=0.0, help="CutMix probability")
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
        mixup_prob=cfg.mixup_prob,
        cutmix_prob=cfg.cutmix_prob,
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
    if args.config:
        cfg = TrainingConfig.load(args.config, env_prefix=args.env_prefix)
    else:
        cfg = TrainingConfig(data_root=args.data_root)

    cfg.data_root = args.data_root
    cfg.batch_size = args.batch_size
    cfg.epochs = args.epochs
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.fold = args.fold
    cfg.positive_only = args.positive_only
    cfg.gpus = args.gpus
    cfg.train_crop_size = (
        args.train_depth_window_size,
        args.train_spatial_window_size,
        args.train_spatial_window_size,
    )
    cfg.valid_crop_size = (
        args.valid_depth_window_size,
        args.valid_spatial_window_size,
        args.valid_spatial_window_size,
    )
    cfg.transfer_weights = args.transfer_weights
    cfg.freeze_backbone_epochs = args.freeze_backbone_epochs
    cfg.pin_memory = args.pin_memory
    cfg.prefetch_factor = args.prefetch_factor
    cfg.use_gpu_augment = not args.cpu_augment
    cfg.mixup_prob = args.mixup
    cfg.cutmix_prob = args.cutmix

    train(cfg)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
