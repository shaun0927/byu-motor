from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import zarr

from motor_det.utils.target import build_target_maps, build_target_maps_torch
from motor_det.utils.augment import random_crop_around_point

from .detection_dataset import DetectionDataset
from .mixin import PatchCacheMixin

__all__ = ["InstanceCropDataset", "PositiveOnlyCropDataset"]


class InstanceCropDataset(DetectionDataset, PatchCacheMixin):
    """Crop patches around GT centres with optional negatives."""

    def __init__(
        self,
        zarr_path: Path | str,
        center_xyz: np.ndarray,
        voxel_spacing: float,
        crop_size: Tuple[int, int, int] = (96, 128, 128),
        num_crops: int | None = 64,
        negative_ratio: float = 0.2,
        cache_size: int = 128,
        *,
        use_gpu: bool = True,
        mixup_prob: float = 0.0,
        cutmix_prob: float = 0.0,
        copy_paste_prob: float = 0.0,
        copy_paste_limit: int = 1,
    ) -> None:
        DetectionDataset.__init__(
            self,
            use_gpu=use_gpu,
            mixup_prob=mixup_prob,
            cutmix_prob=cutmix_prob,
            copy_paste_prob=copy_paste_prob,
            copy_paste_limit=copy_paste_limit,
        )
        PatchCacheMixin.__init__(self, cache_size=cache_size)
        self.vol = zarr.open(zarr_path, mode="r")
        self.centers = center_xyz.astype(np.float32) / voxel_spacing
        self.spacing = float(voxel_spacing)
        self.crop_size = crop_size
        if num_crops is None:
            n_pos = len(self.centers)
            self.num_crops = int(max(n_pos * (1 + negative_ratio), 1))
        else:
            self.num_crops = int(num_crops)
        self.neg_ratio = float(negative_ratio)

    def __len__(self) -> int:
        return self.num_crops

    # sampling utilities -------------------------------------------------
    def _sample_positive(self):
        idx = np.random.randint(len(self.centers))
        ctr = self.centers[idx]
        patch, start = random_crop_around_point(self.vol, ctr, self.crop_size)
        return patch, start

    def _sample_negative(self):
        Z, Y, X = self.vol.shape
        D, H, W = self.crop_size
        z0 = np.random.randint(0, max(1, Z - D + 1))
        y0 = np.random.randint(0, max(1, Y - H + 1))
        x0 = np.random.randint(0, max(1, X - W + 1))
        start = (z0, y0, x0)
        patch = self._load_patch_cached(self.vol, start, self.crop_size)
        return patch, start

    def _sample_patch_maps(self):
        use_negative = np.random.rand() < self.neg_ratio or len(self.centers) == 0
        if use_negative:
            patch, start = self._sample_negative()
        else:
            patch, start = self._sample_positive()

        z0, y0, x0 = start
        z1, y1, x1 = z0 + self.crop_size[0], y0 + self.crop_size[1], x0 + self.crop_size[2]

        mask = (
            (self.centers[:, 2] >= z0)
            & (self.centers[:, 2] < z1)
            & (self.centers[:, 1] >= y0)
            & (self.centers[:, 1] < y1)
            & (self.centers[:, 0] >= x0)
            & (self.centers[:, 0] < x1)
        )
        centers_local = self.centers[mask] - np.array([x0, y0, z0], np.float32)

        if self.use_gpu:
            device = torch.device("cuda")
            centers_t = torch.as_tensor(centers_local, dtype=torch.float32, device=device)
            cls_map_t, off_map_t = build_target_maps_torch(
                centers_t, crop_size=self.crop_size, stride=2, device=device
            )
            patch_t = torch.as_tensor(patch, dtype=torch.float32, device=device)
            return patch_t, cls_map_t, off_map_t, centers_t
        else:
            cls_map_np, off_map_np = build_target_maps(
                centers_local.astype(np.float32), crop_size=self.crop_size, stride=2
            )
            return patch, cls_map_np, off_map_np, centers_local

    # main API -----------------------------------------------------------
    def __getitem__(self, idx: int):
        patch, cls_map, off_map, centers = self._sample_patch_maps()

        patch, cls_map, off_map, centers = self.apply_augmentations(
            patch, cls_map, off_map, centers, self._sample_patch_maps
        )

        return self.convert_to_dict(patch, cls_map, off_map, centers)


class PositiveOnlyCropDataset(InstanceCropDataset):
    def __init__(self, *args, **kw):
        kw["negative_ratio"] = 0.0
        super().__init__(*args, **kw)

