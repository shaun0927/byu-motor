# ======================================================================
#  aggregation.py – patch-prob ➜ volume-prob + 3-D NMS
#  * FP16 입력 친화 · Gaussian kernel LRU-cache *
# ======================================================================
from __future__ import annotations
from functools import lru_cache
from typing import List, Tuple

import numpy as np
from scipy.ndimage import maximum_filter

__all__ = [
    "gaussian_kernel",
    "fuse_patches",
    "nms_3d",
    "aggregate_and_peaks",
    "aggregate_probability",
]

# ------------------------------------------------------------------ #
# 1. 3-D Gaussian weighting kernel  (cached ❶)
# ------------------------------------------------------------------ #
@lru_cache(maxsize=16)
def gaussian_kernel(size: Tuple[int, int, int],
                    sigma: float | Tuple[float, float, float]) -> np.ndarray:
    """
    Parameters
    ----------
    size  : (dz, dy, dx) patch shape
    sigma : float or (σz, σy, σx) in voxels
    Returns
    -------
    ker   : float32, peak=1, same shape as *size*
    """
    if isinstance(sigma, (int, float)):
        sigma = (sigma, sigma, sigma)

    z, y, x = [np.linspace(-(s - 1) / 2, (s - 1) / 2, s, dtype=np.float32)
               for s in size]
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    ker = np.exp(-(zz ** 2) / (2 * sigma[0] ** 2)
                 -(yy ** 2) / (2 * sigma[1] ** 2)
                 -(xx ** 2) / (2 * sigma[2] ** 2))
    ker /= ker.max()
    return ker.astype(np.float32)            # always float32


# ------------------------------------------------------------------ #
# 2. patch-prob list → full-volume prob (dtype-aware)
# ------------------------------------------------------------------ #
def fuse_patches(prob_list: List[np.ndarray],
                 starts:    List[Tuple[int, int, int]],
                 vol_shape: Tuple[int, int, int],
                 weight_ker: np.ndarray | None = None,
                 *,
                 out_dtype = np.float32,
                 ) -> np.ndarray:
    """
    Parameters
    ----------
    prob_list : list[(dz,dy,dx)] foreground prob volumes (any fp dtype)
    starts    : list[(z0,y0,x0)] global start indices
    vol_shape : (Z,Y,X) – full tomogram shape
    weight_ker: same shape as patch (float32).  If None → all-ones.
    out_dtype : dtype of returned volume (default float32)
    """
    Z, Y, X = vol_shape
    # 내부 누적은 float32 로 – FP16 누적 오류 방지
    acc  = np.zeros((Z, Y, X), np.float32)
    wmap = np.zeros_like(acc)

    if weight_ker is None:
        weight_ker = np.ones_like(prob_list[0], np.float32)
    else:
        weight_ker = weight_ker.astype(np.float32, copy=False)

    for prob, (z0, y0, x0) in zip(prob_list, starts):
        prob32 = prob.astype(np.float32, copy=False)  # convert if FP16
        z1, y1, x1 = np.array([z0, y0, x0]) + prob32.shape
        acc [z0:z1, y0:y1, x0:x1] += prob32 * weight_ker
        wmap[z0:z1, y0:y1, x0:x1] += weight_ker

    np.maximum(wmap, 1e-6, out=wmap)           # div-0 guard
    fused = acc / wmap
    return fused.astype(out_dtype, copy=False)


# ------------------------------------------------------------------ #
# 3. simple 3-D NMS (Chebyshev radius)
# ------------------------------------------------------------------ #
def nms_3d(prob_vol: np.ndarray,
           thr: float = 0.3,
           radius: int = 3,
           top_k: int | None = 1
           ) -> List[Tuple[int, int, int, float]]:
    """
    Returns list[(x, y, z, conf)] sorted by conf desc.
    """
    if prob_vol.max() < thr:
        return []

    mx   = maximum_filter(prob_vol, size=2 * radius + 1, mode="constant")
    mask = (prob_vol == mx) & (prob_vol >= thr)

    zz, yy, xx = np.where(mask)
    conf = prob_vol[zz, yy, xx]
    order = np.argsort(-conf)
    if top_k is not None:
        order = order[:top_k]

    return [(int(x), int(y), int(z), float(c))
            for z, y, x, c in zip(zz[order], yy[order], xx[order], conf[order])]


# ------------------------------------------------------------------ #
# 4. convenience wrapper (patches → peak list)
# ------------------------------------------------------------------ #
def aggregate_and_peaks(prob_list: List[np.ndarray],
                        starts:    List[Tuple[int, int, int]],
                        vol_shape: Tuple[int, int, int],
                        *,
                        sigma: float = 4.0,
                        thr: float = 0.3,
                        nms_r: int = 3,
                        top_k: int | None = 1) -> List[Tuple[int, int, int, float]]:
    ker   = gaussian_kernel(tuple(prob_list[0].shape), sigma)
    fused = fuse_patches(prob_list, starts, vol_shape, ker)
    return nms_3d(fused, thr=thr, radius=nms_r, top_k=top_k)


# ------------------------------------------------------------------ #
# 5. backward-compat ‒ same signature as old train.py
# ------------------------------------------------------------------ #
def aggregate_probability(prob_list, starts, vol_shape, sigma: float = 4.0):
    ker = gaussian_kernel(tuple(prob_list[0].shape), sigma)
    return fuse_patches(prob_list, starts, vol_shape, ker)
