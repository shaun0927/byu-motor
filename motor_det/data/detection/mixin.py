import numpy as np
import torch
from collections import OrderedDict
from typing import Tuple

__all__ = [
    "_apply_flip_np",
    "_apply_flip_torch",
    "ObjectDetectionMixin",
    "PatchCacheMixin",
]


def _apply_flip_np(centers: np.ndarray, axes: tuple[int, ...], shape: Tuple[int, int, int]) -> np.ndarray:
    if centers.size == 0 or not axes:
        return centers
    out = centers.copy()
    axis_map = {0: 2, 1: 1, 2: 0}
    for ax in axes:
        coord_ax = axis_map[ax]
        out[:, coord_ax] = shape[ax] - 1 - out[:, coord_ax]
    return out


def _apply_flip_torch(centers: torch.Tensor, axes: tuple[int, ...], shape: Tuple[int, int, int]) -> torch.Tensor:
    if centers.numel() == 0 or not axes:
        return centers
    out = centers.clone()
    axis_map = {0: 2, 1: 1, 2: 0}
    for ax in axes:
        coord_ax = axis_map[ax]
        out[:, coord_ax] = shape[ax] - 1 - out[:, coord_ax]
    return out


class ObjectDetectionMixin:
    """Mixin converting tensors/arrays to a unified dict."""

    spacing: float = 1.0

    def convert_to_dict(self, image, cls_map, off_map, centers):
        if torch.is_tensor(image):
            img_t = (image / 255.0).unsqueeze(0)
            cls_t = cls_map
            off_t = off_map
            ctrs_t = centers if torch.is_tensor(centers) else torch.as_tensor(centers, dtype=torch.float32, device=image.device)
        else:
            img_t = torch.as_tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
            cls_t = torch.as_tensor(cls_map)
            off_t = torch.as_tensor(off_map)
            ctrs_t = torch.as_tensor(centers, dtype=torch.float32)
        return {
            "image": img_t,
            "cls": cls_t,
            "offset": off_t,
            "centers_Å": ctrs_t * self.spacing,
            "spacing_Å_per_voxel": self.spacing,
        }


class PatchCacheMixin:
    """Provide simple LRU caching for patches."""

    def __init__(self, cache_size: int = 128) -> None:
        self._cache_size = int(cache_size)
        self._patch_cache: OrderedDict[Tuple[int, int, int], np.ndarray] = OrderedDict()

    def _load_patch_cached(self, vol, start: Tuple[int, int, int], size: Tuple[int, int, int]) -> np.ndarray:
        z0, y0, x0 = start
        key = (z0, y0, x0)
        if key in self._patch_cache:
            self._patch_cache.move_to_end(key)
            return self._patch_cache[key]
        D, H, W = size
        patch = np.asarray(vol[z0 : z0 + D, y0 : y0 + H, x0 : x0 + W], dtype=np.uint8)
        self._patch_cache[key] = patch
        if len(self._patch_cache) > self._cache_size:
            self._patch_cache.popitem(last=False)
        return patch
