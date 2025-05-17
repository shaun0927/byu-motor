# motor_det/data/dataset.py
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr

from motor_det.utils.target import build_target_maps
from motor_det.utils.augment import random_flip3d, random_crop_around_point


class MotorTrainDataset(Dataset):
    """
    BYU Motor – 학습용 랜덤 Crop + Flip 증강 (Lazy Zarr I/O 버전)

    반환 dict:
      image     : FloatTensor [1, D, H, W]
      cls       : FloatTensor [1, D/2, H/2, W/2]    ← 이미 채널축 포함
      offset    : FloatTensor [3, D/2, H/2, W/2]
      centers_Å : FloatTensor [K, 3]  (가변 길이)
    """

    def __init__(
        self,
        zarr_path: Path,
        center_xyz: np.ndarray,  # shape (N,3) in Å
        voxel_spacing: float,  # Å/voxel
        crop_size: Tuple[int, int, int] = (96, 128, 128),
        negative_ratio: float = 0.5,
    ):
        self.vol = zarr.open(zarr_path, mode="r")
        self.centers = center_xyz.astype(np.float32) / voxel_spacing
        self.spacing = float(voxel_spacing)
        self.crop_size = crop_size
        self.neg_ratio = negative_ratio

    def __len__(self):
        return max(2 * len(self.centers), 64)

    def __getitem__(self, idx):
        D, H, W = self.crop_size
        Z, Y, X = self.vol.shape

        # 1) positive / negative crop 선택
        if np.random.rand() > self.neg_ratio and len(self.centers) > 0:
            ctr = self.centers[np.random.randint(len(self.centers))]
            z0 = int(np.clip(ctr[2] - D // 2, 0, Z - D))
            y0 = int(np.clip(ctr[1] - H // 2, 0, Y - H))
            x0 = int(np.clip(ctr[0] - W // 2, 0, X - W))
        else:
            z0 = np.random.randint(0, Z - D + 1)
            y0 = np.random.randint(0, Y - H + 1)
            x0 = np.random.randint(0, X - W + 1)
        z1, y1, x1 = z0 + D, y0 + H, x0 + W

        # 2) 패치 읽어서 정규화
        patch = np.asarray(self.vol[z0:z1, y0:y1, x0:x1], dtype=np.uint8)

        # 3) GT 필터 & 타겟 맵 생성
        mask = (
            (self.centers[:, 2] >= z0)
            & (self.centers[:, 2] < z1)
            & (self.centers[:, 1] >= y0)
            & (self.centers[:, 1] < y1)
            & (self.centers[:, 0] >= x0)
            & (self.centers[:, 0] < x1)
        )
        centers_local = self.centers[mask] - np.array([x0, y0, z0], dtype=np.float32)
        cls_map, off_map = build_target_maps(
            centers_local,
            crop_size=self.crop_size,
            stride=2,
        )
        # cls_map shape == (1, D', H', W')

        # 4) 3축 flip 증강
        patch, cls_map, off_map = random_flip3d(patch, cls_map, off_map)

        # 5) ndarray → torch.Tensor
        #    image: (D,H,W) → [1,D,H,W]
        img_t = torch.from_numpy(patch.astype(np.float32) / 255.0).unsqueeze(0)
        #    cls_map: already (1,D',H',W'), 그대로 tensor
        cls_t = torch.from_numpy(cls_map.astype(np.float32))
        #    off_map: (3,D',H',W')
        off_t = torch.from_numpy(off_map.astype(np.float32))
        #    centers: (K,3)
        centers_A = torch.from_numpy((centers_local * self.spacing).astype(np.float32))

        return {
            "image": img_t,  # [1,D,H,W]
            "cls": cls_t,  # [1,D',H',W']
            "offset": off_t,  # [3,D',H',W']
            "centers_Å": centers_A,  # [K,3]
        }


class MotorInstanceCropDataset(Dataset):
    """Dataset that samples crops around ground-truth centers."""

    def __init__(
        self,
        zarr_path: Path,
        center_xyz: np.ndarray,
        voxel_spacing: float,
        crop_size: Tuple[int, int, int] = (96, 128, 128),
        num_crops: int = 64,
        negative_ratio: float = 0.2,
    ) -> None:
        self.vol = zarr.open(zarr_path, mode="r")
        self.centers = center_xyz.astype(np.float32) / voxel_spacing
        self.spacing = float(voxel_spacing)
        self.crop_size = crop_size
        self.num_crops = int(num_crops)
        self.neg_ratio = float(negative_ratio)

    def __len__(self) -> int:
        return self.num_crops

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
        z1, y1, x1 = z0 + D, y0 + H, x0 + W
        patch = np.asarray(self.vol[z0:z1, y0:y1, x0:x1], dtype=np.uint8)
        return patch, (z0, y0, x0)

    def __getitem__(self, idx):
        use_negative = np.random.rand() < self.neg_ratio or len(self.centers) == 0
        if use_negative:
            patch, start = self._sample_negative()
        else:
            patch, start = self._sample_positive()

        z0, y0, x0 = start
        z1, y1, x1 = (
            z0 + self.crop_size[0],
            y0 + self.crop_size[1],
            x0 + self.crop_size[2],
        )

        mask = (
            (self.centers[:, 2] >= z0)
            & (self.centers[:, 2] < z1)
            & (self.centers[:, 1] >= y0)
            & (self.centers[:, 1] < y1)
            & (self.centers[:, 0] >= x0)
            & (self.centers[:, 0] < x1)
        )
        centers_local = self.centers[mask] - np.array([x0, y0, z0], np.float32)

        cls_map, off_map = build_target_maps(
            centers_local,
            crop_size=self.crop_size,
            stride=2,
        )

        patch, cls_map, off_map = random_flip3d(patch, cls_map, off_map)

        img_t = torch.from_numpy(patch.astype(np.float32) / 255.0).unsqueeze(0)
        cls_t = torch.from_numpy(cls_map.astype(np.float32))
        off_t = torch.from_numpy(off_map.astype(np.float32))
        centers_A = torch.from_numpy((centers_local * self.spacing).astype(np.float32))

        return {
            "image": img_t,
            "cls": cls_t,
            "offset": off_t,
            "centers_Å": centers_A,
        }


class BackgroundRandomCropDataset(Dataset):
    """Simple negative crop dataset used with ``ConcatDataset``."""

    def __init__(self, zarr_path: Path, crop_size: Tuple[int, int, int] = (96, 128, 128), num_crops: int = 16) -> None:
        self.vol = zarr.open(zarr_path, mode="r")
        self.crop_size = crop_size
        self.num_crops = int(num_crops)

    def __len__(self) -> int:
        return self.num_crops

    def __getitem__(self, idx):
        Z, Y, X = self.vol.shape
        D, H, W = self.crop_size
        z0 = np.random.randint(0, max(1, Z - D + 1))
        y0 = np.random.randint(0, max(1, Y - H + 1))
        x0 = np.random.randint(0, max(1, X - W + 1))
        z1, y1, x1 = z0 + D, y0 + H, x0 + W
        patch = np.asarray(self.vol[z0:z1, y0:y1, x0:x1], dtype=np.uint8)
        cls_map = np.zeros((1, D // 2, H // 2, W // 2), np.float32)
        off_map = np.zeros((3, D // 2, H // 2, W // 2), np.float32)

        patch, cls_map, off_map = random_flip3d(patch, cls_map, off_map)

        img_t = torch.from_numpy(patch.astype(np.float32) / 255.0).unsqueeze(0)
        cls_t = torch.from_numpy(cls_map)
        off_t = torch.from_numpy(off_map)

        return {
            "image": img_t,
            "cls": cls_t,
            "offset": off_t,
            "centers_Å": torch.empty((0, 3), dtype=torch.float32),
        }
