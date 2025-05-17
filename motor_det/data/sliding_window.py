# motor_det/data/sliding_window.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
from collections import OrderedDict

import torch

import numpy as np
import zarr
from torch.utils.data import Dataset

__all__ = ["SlidingWindowDataset", "compute_tiles"]


def compute_tiles(
    vol_shape: Tuple[int, int, int],
    win: Tuple[int, int, int] = (192, 128, 128),
    stride: Tuple[int, int, int] = (96, 64, 64),
) -> List[Tuple[slice, slice, slice]]:
    """Return list[ (z-slice, y-slice, x-slice) ] that covers the volume."""
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


class SlidingWindowDataset(Dataset):
    """
    Lazily reads patches from a Zarr tomogram.
    """

    def __init__(
        self,
        zarr_path: Path | str,
        window: Tuple[int, int, int] = (192, 128, 128),
        stride: Tuple[int, int, int] = (96, 64, 64),
        dtype=np.float32,
        *,
        cache_size: int = 128,
    ):
        self.store = zarr.open(zarr_path, mode="r")  # no copy ✓
        self.win = window  # <─★ win 로 한 번 더 보관
        self.window = window
        self.stride = stride
        self.dtype = dtype
        self._cache_size = int(cache_size)
        self._patch_cache: OrderedDict[Tuple[int, int, int], np.ndarray] = OrderedDict()

    @torch.cached_property
    def tiles(self) -> List[Tuple[slice, slice, slice]]:
        return compute_tiles(self.store.shape, self.window, self.stride)

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
        # pad if needed (right / bottom / back edge)
        dz, dy, dx = self.window
        pad = (
            (0, dz - patch.shape[0]),
            (0, dy - patch.shape[1]),
            (0, dx - patch.shape[2]),
        )
        if any(p[1] > 0 for p in pad):
            patch = np.pad(patch, pad, mode="constant")
        return patch[None], (tz.start, ty.start, tx.start)  # (C=1, D,H,W), origin
