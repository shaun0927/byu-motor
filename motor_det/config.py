
from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Type, TypeVar

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

from motor_det.utils.voxel import DEFAULT_TEST_SPACING

T = TypeVar("T")


def _parse_file(path: Path) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r") as f:
        if path.suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise ImportError("PyYAML is required for YAML configuration files")
            data = yaml.safe_load(f) or {}
        else:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a mapping at the top level")
    return data


def _apply_env(data: dict[str, Any], cls: Type[T], prefix: str | None) -> None:
    if not prefix:
        return
    prefix = prefix.upper()
    for field in fields(cls):
        env_key = f"{prefix}{field.name.upper()}"
        if env_key in os.environ:
            val = os.environ[env_key]
            try:
                data[field.name] = ast.literal_eval(val)
            except Exception:
                data[field.name] = val


def _load(cls: Type[T], path: str | Path | None, env_prefix: str | None) -> T:
    data: dict[str, Any] = {}
    if path is not None:
        data.update(_parse_file(Path(path)))
    _apply_env(data, cls, env_prefix)
    return cls(**data)


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    data_root: str
    fold: int = 0
    batch_size: int = 2
    num_workers: int = 12
    persistent_workers: bool = True
    positive_only: bool = True
    train_crop_size: tuple[int, int, int] = (96, 128, 128)
    valid_crop_size: tuple[int, int, int] = (192, 128, 128)
    pin_memory: bool = False
    prefetch_factor: int | None = 2
    use_gpu_augment: bool = True
    valid_use_gpu_augment: bool | None = False
    mixup_prob: float = 0.0
    cutmix_prob: float = 0.0
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    transfer_weights: str | None = None
    freeze_backbone_epochs: int = 0
    gpus: int = 1
    nms_algorithm: str = "vectorized"
    nms_switch_thr: int = 1500
    prob_thr: float = 0.6
    max_steps: int | None = None
    limit_val_batches: float | int = 1.0
    val_check_interval: float | int = 1.0
    num_sanity_val_steps: int = 0
    focal_gamma: float = 2.0
    pos_weight_clip: float = 5.0

    @classmethod
    def load(cls, path: str | Path | None = None, *, env_prefix: str | None = "BYU_TRAIN_") -> "TrainingConfig":
        return _load(cls, path, env_prefix)


@dataclass
class InferenceConfig:
    """Configuration for inference."""

    weights: str
    data_root: str
    out_csv: str
    win_d: int = 192
    win_h: int = 128
    win_w: int = 128
    stride_d: int = 96
    stride_h: int = 64
    stride_w: int = 64
    stride_head: int = 2
    batch: int = 1
    num_workers: int = 4

    # optional sliding window tiling
    num_tiles_d: int | None = None
    num_tiles_h: int | None = None
    num_tiles_w: int | None = None
    tile_xy: int | None = None

    prob_thr: float = 0.6
    sigma: float = 60.0
    iou_thr: float = 0.25
    default_spacing: float = DEFAULT_TEST_SPACING
    early_exit: float | None = None

    @classmethod
    def load(cls, path: str | Path | None = None, *, env_prefix: str | None = "BYU_INFER_") -> "InferenceConfig":
        return _load(cls, path, env_prefix)

__all__ = ["TrainingConfig", "InferenceConfig"]

