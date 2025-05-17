# motor_det/data/dataset.py
from pathlib import Path
from typing import Tuple
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr

from motor_det.utils.target import build_target_maps, build_target_maps_torch
from motor_det.utils.augment import (
    random_flip3d,
    random_crop_around_point,
    random_erase3d,
    random_gaussian_noise,
    random_flip3d_torch,
    random_erase3d_torch,
    random_gaussian_noise_torch,
)


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
        cache_size: int = 128,
    ):
        self.vol = zarr.open(zarr_path, mode="r")
        self.centers = center_xyz.astype(np.float32) / voxel_spacing
        self.spacing = float(voxel_spacing)
        self.crop_size = crop_size
        self.neg_ratio = negative_ratio
        self._cache_size = int(cache_size)
        # simple ordered dict for manual LRU cache
        self._patch_cache: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()

    def _load_patch_cached(self, z0: int, y0: int, x0: int) -> np.ndarray:
        key = (z0, y0, x0)
        if key in self._patch_cache:
            self._patch_cache.move_to_end(key)
            return self._patch_cache[key]

        D, H, W = self.crop_size
        z1, y1, x1 = z0 + D, y0 + H, x0 + W
        patch = np.asarray(self.vol[z0:z1, y0:y1, x0:x1], dtype=np.uint8)
        self._patch_cache[key] = patch
        if len(self._patch_cache) > self._cache_size:
            self._patch_cache.popitem(last=False)
        return patch

    def __len__(self):
        # dataset length proportional to number of positive centres
        n_pos = len(self.centers)
        n_total = int(n_pos * (1 + self.neg_ratio))
        return max(n_total, 64)

    def __getitem__(self, idx):
        D, H, W = self.crop_size
        Z, Y, X = self.vol.shape

        # 1) positive / negative crop 선택
        use_negative = np.random.rand() < self.neg_ratio or len(self.centers) == 0
        if use_negative:
            z0 = np.random.randint(0, Z - D + 1)
            y0 = np.random.randint(0, Y - H + 1)
            x0 = np.random.randint(0, X - W + 1)
        else:
            ctr = self.centers[np.random.randint(len(self.centers))]
            z0 = int(np.clip(ctr[2] - D // 2, 0, Z - D))
            y0 = int(np.clip(ctr[1] - H // 2, 0, Y - H))
            x0 = int(np.clip(ctr[0] - W // 2, 0, X - W))
        z1, y1, x1 = z0 + D, y0 + H, x0 + W

        # 2) 패치 읽어서 정규화 (LRU cache 활용)
        patch = self._load_patch_cached(z0, y0, x0).copy()

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        centers_t = torch.from_numpy(centers_local.astype(np.float32)).to(device)
        cls_map_t, off_map_t = build_target_maps_torch(
            centers_t,
            crop_size=self.crop_size,
            stride=2,
            device=device,
        )
        # cls_map shape == (1, D', H', W')

        # 4) 3축 flip + optional erase/noise augmentation
        patch_t = torch.from_numpy(patch.astype(np.float32)).to(device)
        patch_t, cls_map_t, off_map_t = random_flip3d_torch(patch_t, cls_map_t, off_map_t)
        patch_t = random_erase3d_torch(patch_t)
        patch_t = random_gaussian_noise_torch(patch_t)
        img_t = (patch_t / 255.0).unsqueeze(0)
        centers_A = centers_t * self.spacing

        return {
            "image": img_t,  # [1,D,H,W]
            "cls": cls_map_t,
            "offset": off_map_t,
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
        num_crops: int | None = 64,
        negative_ratio: float = 0.2,
    ) -> None:
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        centers_t = torch.from_numpy(centers_local.astype(np.float32)).to(device)
        cls_map, off_map = build_target_maps_torch(
            centers_t, crop_size=self.crop_size, stride=2, device=device
        )

        patch_t = torch.from_numpy(patch.astype(np.float32)).to(device)
        patch_t, cls_map, off_map = random_flip3d_torch(patch_t, cls_map, off_map)
        patch_t = random_erase3d_torch(patch_t)
        patch_t = random_gaussian_noise_torch(patch_t)
        img_t = (patch_t / 255.0).unsqueeze(0)
        centers_A = centers_t * self.spacing

        return {
            "image": img_t,
            "cls": cls_map,
            "offset": off_map,
            "centers_Å": centers_A,
        }
    
class PositiveOnlyCropDataset(MotorInstanceCropDataset):
    """Dataset that samples only positive crops around ground-truth centers."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["negative_ratio"] = 0.0
        super().__init__(*args, **kwargs)

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls_map = torch.zeros((1, D // 2, H // 2, W // 2), dtype=torch.float32, device=device)
        off_map = torch.zeros((3, D // 2, H // 2, W // 2), dtype=torch.float32, device=device)

        patch_t = torch.from_numpy(patch.astype(np.float32)).to(device)
        patch_t, cls_map, off_map = random_flip3d_torch(patch_t, cls_map, off_map)
        patch_t = random_erase3d_torch(patch_t)
        patch_t = random_gaussian_noise_torch(patch_t)
        img_t = (patch_t / 255.0).unsqueeze(0)
        
        return {
            "image": img_t,
            "cls": cls_map,
            "offset": off_map,
            "centers_Å": torch.empty((0, 3), dtype=torch.float32),
        }

class MotorPositiveCropDataset(Dataset):
    """Dataset that returns one crop around each GT center (positive only)."""

    def __init__(
        self,
        zarr_path: Path,
        center_xyz: np.ndarray,
        voxel_spacing: float,
        crop_size: Tuple[int, int, int] = (96, 128, 128),
        jitter: int = 0,
    ) -> None:
        self.vol = zarr.open(zarr_path, mode="r")
        # voxel coordinates of ground-truth centers
        self.pos_centers = center_xyz.astype(np.float32) / voxel_spacing
        self.spacing = float(voxel_spacing)
        self.crop_size = crop_size
        self.jitter = int(jitter)

    def __len__(self) -> int:
        return len(self.pos_centers)

    def __getitem__(self, idx: int):
        ctr = self.pos_centers[idx]

        patch, start = random_crop_around_point(
            self.vol, ctr, self.crop_size, jitter=self.jitter
        )
        z0, y0, x0 = start
        z1, y1, x1 = (
            z0 + self.crop_size[0],
            y0 + self.crop_size[1],
            x0 + self.crop_size[2],
        )

        mask = (
            (self.pos_centers[:, 2] >= z0)
            & (self.pos_centers[:, 2] < z1)
            & (self.pos_centers[:, 1] >= y0)
            & (self.pos_centers[:, 1] < y1)
            & (self.pos_centers[:, 0] >= x0)
            & (self.pos_centers[:, 0] < x1)
        )
        centers_local = self.pos_centers[mask] - np.array([x0, y0, z0], np.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        centers_t = torch.from_numpy(centers_local.astype(np.float32)).to(device)
        cls_map, off_map = build_target_maps_torch(
            centers_t, crop_size=self.crop_size, stride=2, device=device
        )

        patch_t = torch.from_numpy(patch.astype(np.float32)).to(device)
        patch_t, cls_map, off_map = random_flip3d_torch(patch_t, cls_map, off_map)
        patch_t = random_erase3d_torch(patch_t)
        patch_t = random_gaussian_noise_torch(patch_t)
        img_t = (patch_t / 255.0).unsqueeze(0)
        centers_A = centers_t * self.spacing

        return {
            "image": img_t,
            "cls": cls_map,
            "offset": off_map,
            "centers_Å": centers_A,
        }
