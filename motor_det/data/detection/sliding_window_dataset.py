from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
from collections import OrderedDict

import numpy as np
import torch
from functools import cached_property
import zarr
from torch.utils.data import Dataset

from .mixin import ObjectDetectionMixin, PatchCacheMixin
from ...utils.tile import compute_better_tiles_with_num_tiles

__all__ = ["SlidingWindowDataset", "compute_tiles", "compute_tiles_with_num_tiles"]


def compute_tiles(
    vol_shape: Tuple[int, int, int],
    win: Tuple[int, int, int] = (192, 128, 128),
    stride: Tuple[int, int, int] = (96, 64, 64),
) -> List[Tuple[slice, slice, slice]]:
    """Vectorised sliding window tile computation."""

    dz, dy, dx = win
    sz, sy, sx = stride
    Z, Y, X = vol_shape

    z_starts = np.arange(0, max(1, Z - dz + 1), sz, dtype=int)
    y_starts = np.arange(0, max(1, Y - dy + 1), sy, dtype=int)
    x_starts = np.arange(0, max(1, X - dx + 1), sx, dtype=int)

    zz, yy, xx = np.meshgrid(z_starts, y_starts, x_starts, indexing="ij")
    zz = zz.ravel()
    yy = yy.ravel()
    xx = xx.ravel()

    z1 = np.minimum(zz + dz, Z)
    y1 = np.minimum(yy + dy, Y)
    x1 = np.minimum(xx + dx, X)

    return [
        (slice(int(z0), int(z1_)), slice(int(y0), int(y1_)), slice(int(x0), int(x1_)))
        for z0, z1_, y0, y1_, x0, x1_ in zip(zz, z1, yy, y1, xx, x1)
    ]


def compute_tiles_with_num_tiles(
    vol_shape: Tuple[int, int, int],
    win: Tuple[int, int, int] = (192, 128, 128),
    num_tiles: Tuple[int, int, int] = (1, 9, 9),
) -> List[Tuple[slice, slice, slice]]:
    return list(compute_better_tiles_with_num_tiles(vol_shape, win, num_tiles))


class SlidingWindowDataset(PatchCacheMixin, Dataset, ObjectDetectionMixin):
    """Lazily reads patches from a Zarr tomogram."""

    def __init__(
        self,
        zarr_path: Path | str,
        window: Tuple[int, int, int] = (192, 128, 128),
        stride: Tuple[int, int, int] = (96, 64, 64),
        dtype=np.float32,
        *,
        cache_size: int = 128,
        num_tiles: Tuple[int, int, int] | None = None,
        voxel_spacing: float = 1.0,
        preload_volume: bool = False,
    ) -> None:
        self.store = zarr.open(zarr_path, mode="r")
        if preload_volume:
            self.store = np.asarray(self.store).astype(dtype)
        self.win = window
        self.window = window
        self.stride = stride
        self.num_tiles = num_tiles
        self.dtype = dtype
        self.spacing = float(voxel_spacing)
        self._cache_size = int(cache_size)
        self._patch_cache: OrderedDict[Tuple[int, int, int], np.ndarray] = OrderedDict()

    @cached_property
    def tiles(self) -> List[Tuple[slice, slice, slice]]:
        if self.num_tiles is None:
            return compute_tiles(self.store.shape, self.window, self.stride)
        return compute_tiles_with_num_tiles(self.store.shape, self.window, self.num_tiles)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tz, ty, tx = self.tiles[idx]
        # Mixin이 제공하는 LRU 캐시 사용
        start = (tz.start, ty.start, tx.start)
        patch = self._load_patch_cached(self.store, start, self.window).astype(self.dtype)

        dz, dy, dx = self.window
        pad = (
            (0, dz - patch.shape[0]),
            (0, dy - patch.shape[1]),
            (0, dx - patch.shape[2]),
        )
        if any(p[1] > 0 for p in pad):
            patch = np.pad(patch, pad, mode="constant")

        return (
            self.convert_to_dict(torch.from_numpy(patch), None, None, np.empty((0, 3), np.float32))
            | {"origin": start}
        )

