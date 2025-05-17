# =====================================================================
#  src/datasets/motor_dataset.py    –  stride-free, cached, NHWC-free
# =====================================================================
from __future__ import annotations
from pathlib import Path
from typing  import Dict, List, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr

# ──────────────── helpers ────────────────────────────────────────────
def coords_to_heat(
    coords: np.ndarray, roi: Tuple[int, int, int], sigma: float = 4.0
) -> np.ndarray:
    """(K,3)[zyx] → (1,dz,dy,dx) heat-map ∈ [0,1]."""
    heat = np.zeros((1, *roi), np.float32)
    if coords.size == 0:
        return heat
    zz, yy, xx = np.meshgrid(*(np.arange(s) for s in roi), indexing="ij")
    for zc, yc, xc in coords:
        d2 = (zz - zc) ** 2 + (yy - yc) ** 2 + (xx - xc) ** 2
        heat[0] = np.maximum(heat[0], np.exp(-0.5 * d2 / sigma**2))
    return heat
# ---------------------------------------------------------------------


class MotorDataset(Dataset):
    """
    Fast Zarr-patch loader (train / val / test)  
    · 모든 모드가 `sub_epochs` 만큼 **무작위 패치**를 뽑는다.
      – val/test 에서는 config 에서 sub_epochs=1 로 두면
        ‘볼륨당 1 패치’ → 지연 없이 빠른 평가가 가능.
    """

    # -----------------------------------------------------------------
    def __init__(
        self,
        df,
        cfg,
        *,
        data_root: str | Path,
        mode: str = "train",
        aug=None,
    ):
        super().__init__()
        self.mode   = mode.lower()
        self.aug    = aug
        self.roi    = np.asarray(getattr(cfg, "roi_size", (96, 96, 96)), int)
        self.sigma  = float(getattr(cfg, "sigma_px", 4.0))
        self.rng    = np.random.default_rng(cfg.seed)
        self.data_root = Path(data_root)

        # -------- volume list ----------------------------------------
        self.items: List[Dict[str, Any]] = []
        for tid, sub in df.groupby("tomo_id", dropna=True):
            tid = str(tid).strip()
            fp  = (self.data_root / f"{tid}.zarr").resolve()
            if not fp.exists():
                raise FileNotFoundError(fp)
            motors = (
                sub[["Motor axis 0", "Motor axis 1", "Motor axis 2"]]
                .to_numpy(float)
            )
            motors = motors[~(motors == -1).any(1)]
            self.items.append(dict(tid=tid, fp=fp, motors=motors))

        # -------- epoch-size 결정 -----------------------------------
        self.sub_ep = int(getattr(cfg, "train_sub_epochs", 1)) \
                      if self.mode == "train" else 1   # ← val/test 은 1

        # -------- per-worker Zarr cache -----------------------------
        self._zf_cache: Dict[Path, zarr.core.Array] = {}

    # -----------------------------------------------------------------
    def __len__(self):
        return len(self.items) * self.sub_ep

    # -----------------------------------------------------------------
    def _open_zarr(self, fp: Path):
        """per-worker LRU cache (≤8 files)."""
        zf = self._zf_cache.get(fp)
        if zf is None:
            if len(self._zf_cache) > 8:
                self._zf_cache.pop(next(iter(self._zf_cache)))
            zf = zarr.open(fp, "r")
            self._zf_cache[fp] = zf
        return zf

    # --------------------- start 좌표 util ---------------------------
    def _clip_start(self, start: np.ndarray, shape: Tuple[int, int, int]):
        """ensure start+roi ≤ shape."""
        max_st = np.maximum(0, np.array(shape) - self.roi)
        return np.clip(start, 0, max_st).astype(int)

    def _rand_start(self, shape, motors):
        Z, Y, X = shape
        dz, dy, dx = self.roi
        if motors.size and self.rng.random() < 0.5:
            c = motors[self.rng.integers(len(motors))] \
                + self.rng.integers(-self.roi // 4, self.roi // 4 + 1)
            start = c - self.roi // 2
        else:
            start = self.rng.integers(
                [0, 0, 0], [max(1, Z - dz), max(1, Y - dy), max(1, X - dx)]
            )
        return self._clip_start(start, shape)

    # -----------------------------------------------------------------
    def __getitem__(self, idx: int):
        vidx  = idx % len(self.items)            # (volume, repetition)
        it    = self.items[vidx]
        zf    = self._open_zarr(it["fp"])
        start = self._rand_start(zf.shape, it["motors"])

        z0, y0, x0 = start
        z1, y1, x1 = start + self.roi
        patch = zf[z0:z1, y0:y1, x0:x1].astype(np.float32) / 255.0   # (dz,dy,dx)
        patch = patch[None]                                          # (1, …)

        if self.mode != "test":
            local = it["motors"] - start
            heat  = coords_to_heat(local, tuple(self.roi), self.sigma)
            fg    = float(local.size > 0)
        else:
            heat = np.zeros_like(patch); fg = 0.0

        sample = dict(
            image     = torch.from_numpy(patch),
            mask      = torch.from_numpy(heat),
            roi_start = torch.as_tensor(start, dtype=torch.int32),
            fg_flag   = torch.tensor(fg, dtype=torch.float32),
            tomo_id   = it["tid"],
        )
        if self.mode == "train" and self.aug:
            sample = self.aug(sample)
        if self.mode == "test":
            sample.pop("mask")
        return sample


# -------------------------- collate ----------------------------------
def tr_collate_fn(batch):
    out: Dict[str, Any] = {}
    for k in batch[0]:
        v0 = batch[0][k]
        out[k] = torch.stack([b[k] for b in batch]) if torch.is_tensor(v0) \
                 else [b[k] for b in batch]
    return out

val_collate_fn = tr_collate_fn
