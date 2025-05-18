from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
from collections import OrderedDict

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from .mixin import ObjectDetectionMixin

__all__ = ["SlidingWindowDataset", "compute_tiles", "compute_tiles_with_num_tiles"]


def compute_tiles(
    vol_shape: Tuple[int, int, int],
    win: Tuple[int, int, int] = (192, 128, 128),
    stride: Tuple[int, int, int] = (96, 64, 64),
) -> List[Tuple[slice, slice, slice]]:
    dz, dy, dx = win
    sz, sy, sx = stride
    Z, Y, X = vol_shape
    tiles = []
    for z0 in range(0, max(1, Z - dz + 1), sz):
        for y0 in range(0, max(1, Y - dy + 1), sy):
            for x0 in range(0, max(1, X - dx + 1), sx):
                z1 = min(z0 + dz, Z)
                y1 = min(y0 + dy, Y)
                x1 = min(x0 + dx, X)
                tiles.append((slice(z0, z1), slice(y0, y1), slice(x0, x1)))
    return tiles


def compute_tiles_with_num_tiles(
    vol_shape: Tuple[int, int, int],
    win: Tuple[int, int, int] = (192, 128, 128),
    num_tiles: Tuple[int, int, int] = (1, 9, 9),
) -> List[Tuple[slice, slice, slice]]:
    dz, dy, dx = win
    nz, ny, nx = num_tiles
    Z, Y, X = vol_shape

    def starts(L: int, w: int, n: int) -> List[int]:
        if n <= 1:
            return [max(0, (L - w) // 2)]
        return [int(round(v)) for v in np.linspace(0, max(0, L - w), n)]

    z_starts = starts(Z, dz, nz)
    y_starts = starts(Y, dy, ny)
    x_starts = starts(X, dx, nx)

    tiles = []
    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                z1 = min(z0 + dz, Z)
                y1 = min(y0 + dy, Y)
                x1 = min(x0 + dx, X)
                tiles.append((slice(z0, z1), slice(y0, y1), slice(x0, x1)))
    return tiles


class SlidingWindowDataset(Dataset, ObjectDetectionMixin):
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
    ) -> None:
        self.store = zarr.open(zarr_path, mode="r")
        self.win = window
        self.window = window
        self.stride = stride
        self.num_tiles = num_tiles
        self.dtype = dtype
        self.spacing = float(voxel_spacing)
        self._cache_size = int(cache_size)
        self._patch_cache: OrderedDict[Tuple[int, int, int], np.ndarray] = OrderedDict()

    @torch.cached_property
    def tiles(self) -> List[Tuple[slice, slice, slice]]:
        if self.num_tiles is None:
            return compute_tiles(self.store.shape, self.window, self.stride)
        return compute_tiles_with_num_tiles(self.store.shape, self.window, self.num_tiles)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tz, ty, tx = self.tiles[idx]
        key = (tz.start, ty.start, tx.start)
        if key in self._patch_cache:
            self._patch_cache.move_to_end(key)
            patch = self._patch_cache[key]
        else:
            patch = self.store[tz, ty, tx].astype(self.dtype)
            self._patch_cache[key] = patch
            if len(self._patch_cache) > self._cache_size:
                self._patch_cache.popitem(last=False)
        dz, dy, dx = self.window
        pad = (
            (0, dz - patch.shape[0]),
            (0, dy - patch.shape[1]),
            (0, dx - patch.shape[2]),
        )
        if any(p[1] > 0 for p in pad):
            patch = np.pad(patch, pad, mode="constant")
        return self.convert_to_dict(torch.from_numpy(patch), None, None, np.empty((0, 3), np.float32)) | {
            "origin": (tz.start, ty.start, tx.start)
        }

