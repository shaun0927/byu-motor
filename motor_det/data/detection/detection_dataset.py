from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from motor_det.utils.augment import (
    random_flip3d,
    random_erase3d,
    random_gaussian_noise,
    random_flip3d_torch,
    random_erase3d_torch,
    random_gaussian_noise_torch,
    mixup3d,
    cutmix3d,
    mixup3d_torch,
    cutmix3d_torch,
)

from .mixin import _apply_flip_np, _apply_flip_torch, ObjectDetectionMixin


class DetectionDataset(Dataset, ObjectDetectionMixin):
    """Base dataset providing augmentation utilities."""

    def __init__(self, *, use_gpu: bool = True, mixup_prob: float = 0.0, cutmix_prob: float = 0.0) -> None:
        self.use_gpu = bool(use_gpu) and torch.cuda.is_available()
        self.mixup_prob = float(mixup_prob)
        self.cutmix_prob = float(cutmix_prob)

    def apply_augmentations(
        self,
        patch,
        cls_map,
        off_map,
        centers,
        sample_fn: Callable[[], Tuple] | None = None,
    ):
        if self.use_gpu:
            patch, cls_map, off_map, axes = random_flip3d_torch(patch, cls_map, off_map, return_axes=True)
            centers = _apply_flip_torch(centers, axes, self.crop_size)
            patch = random_erase3d_torch(patch)
            patch = random_gaussian_noise_torch(patch)
        else:
            patch, cls_map, off_map, axes = random_flip3d(patch, cls_map, off_map, return_axes=True)
            centers = _apply_flip_np(centers, axes, self.crop_size)
            patch = random_erase3d(patch)
            patch = random_gaussian_noise(patch)

        if sample_fn is not None:
            r = np.random.rand()
            if r < self.mixup_prob:
                p2, c2, o2, _ = sample_fn()
                if self.use_gpu:
                    patch, cls_map, off_map = mixup3d_torch(patch, cls_map, off_map, p2, c2, o2)
                else:
                    patch, cls_map, off_map = mixup3d(patch, cls_map, off_map, p2, c2, o2)
            elif r < self.mixup_prob + self.cutmix_prob:
                p2, c2, o2, _ = sample_fn()
                if self.use_gpu:
                    patch, cls_map, off_map = cutmix3d_torch(patch, cls_map, off_map, p2, c2, o2)
                else:
                    patch, cls_map, off_map = cutmix3d(patch, cls_map, off_map, p2, c2, o2)

        return patch, cls_map, off_map, centers

