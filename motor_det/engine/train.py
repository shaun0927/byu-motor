"""
Minimal BYU-Motor training runner
---------------------------------
예시:
python -m motor_det.engine.train ^
       --data_root D:\abs\path\data ^
       --batch_size 2 ^
       --epochs 10
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


# ---------------------------------------------------------------------- #
def train(cfg: TrainingConfig) -> L.Trainer:
    """Train model using the provided configuration."""

    dm = MotorDataModule(
        data_root=cfg.data_root,
        fold=cfg.fold,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.persistent_workers,
        positive_only=cfg.positive_only,
        train_num_instance_crops=cfg.train_num_instance_crops,
        train_num_random_crops=cfg.train_num_random_crops,
        train_include_sliding_dataset=cfg.train_include_sliding_dataset,
        train_crop_size=cfg.train_crop_size,
        valid_crop_size=cfg.valid_crop_size,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor,
        preload_volumes=cfg.preload_volumes,
        use_gpu_augment=cfg.use_gpu_augment,
        valid_use_gpu_augment=cfg.valid_use_gpu_augment,
        mixup_prob=cfg.mixup_prob,
        cutmix_prob=cfg.cutmix_prob,
        copy_paste_prob=cfg.copy_paste_prob,
        copy_paste_limit=cfg.copy_paste_limit,
    )
    dm.setup()

    total_steps = cfg.max_steps or len(dm.train_dataloader()) * cfg.epochs
    model = LitMotorDet(
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        total_steps=total_steps,
        nms_algorithm=cfg.nms_algorithm,
        nms_switch_thr=cfg.nms_switch_thr,
        prob_thr=cfg.prob_thr,
        focal_gamma=cfg.focal_gamma,
        pos_weight_clip=cfg.pos_weight_clip,
    )
    if cfg.transfer_weights:
        ckpt = torch.load(cfg.transfer_weights, map_location="cpu")
        model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)

    callbacks = []
    if cfg.freeze_backbone_epochs > 0:
        callbacks.append(FreezeBackbone(cfg.freeze_backbone_epochs))

    trainer = L.Trainer(
        accelerator="gpu" if cfg.gpus else "cpu",
        devices=cfg.gpus or 1,
        precision="16-mixed",
        max_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        default_root_dir=Path("runs") / f"motor_fold{cfg.fold}",
        log_every_n_steps=50,
        val_check_interval=cfg.val_check_interval,
        limit_val_batches=cfg.limit_val_batches,
        num_sanity_val_steps=cfg.num_sanity_val_steps,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)
    return trainer


# ---------------------------------------------------------------------- #
def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, type=str, help="절대 경로로 지정")
    p.add_argument("--fold", type=int, default=0)

    # dataloader & dataset
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--preload_volumes", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument(
        "--positive_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="only sample patches containing motors",
    )
    p.add_argument("--train_num_instance_crops", type=int, default=128)
    p.add_argument("--train_num_random_crops", type=int, default=0)
    p.add_argument("--train_include_sliding_dataset", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--cpu_augment", action="store_true")
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--cutmix", type=float, default=0.0)
    p.add_argument("--copy_paste", type=float, default=0.0)
    p.add_argument("--copy_paste_limit", type=int, default=1)

    # crop sizes
    p.add_argument("--train_depth", type=int, default=96)
    p.add_argument("--train_spatial", type=int, default=128)
    p.add_argument("--valid_depth", type=int, default=192)
    p.add_argument("--valid_spatial", type=int, default=128)

    # optimisation & schedule
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--limit_val_batches", type=float, default=1.0)
    p.add_argument("--val_check_interval", type=float, default=1.0)
    p.add_argument("--num_sanity_val_steps", type=int, default=0)

    p.add_argument("--prob_thr", type=float, default=0.02)

    # misc
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--transfer_weights", type=str, default=None)
    p.add_argument("--freeze_backbone_epochs", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------- #
def main() -> None:
    args = cli()
    L.seed_everything(42)
    torch.set_float32_matmul_precision("high")

    cfg = TrainingConfig(
        data_root=args.data_root,
        fold=args.fold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        positive_only=args.positive_only,
        train_num_instance_crops=args.train_num_instance_crops,
        train_num_random_crops=args.train_num_random_crops,
        train_include_sliding_dataset=args.train_include_sliding_dataset,
        train_crop_size=(args.train_depth, args.train_spatial, args.train_spatial),
        valid_crop_size=(args.valid_depth, args.valid_spatial, args.valid_spatial),
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        preload_volumes=args.preload_volumes,
        use_gpu_augment=not args.cpu_augment,
        valid_use_gpu_augment=False,
        mixup_prob=args.mixup,
        cutmix_prob=args.cutmix,
        copy_paste_prob=args.copy_paste,
        copy_paste_limit=args.copy_paste_limit,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        transfer_weights=args.transfer_weights,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        gpus=args.gpus,
        prob_thr=args.prob_thr,
        max_steps=args.max_steps,
        limit_val_batches=args.limit_val_batches,
        val_check_interval=args.val_check_interval,
        num_sanity_val_steps=args.num_sanity_val_steps,
    )

    train(cfg)


if __name__ == "__main__":
    main()
