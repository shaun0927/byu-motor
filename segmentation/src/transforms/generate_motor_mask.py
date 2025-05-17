# ======================================================================
#  On-the-fly 3-D motor-mask generator                (channel-first)
#  ------------------------------------------------------------------
#  • dict 입력:
#       ├── image   : (1, Z, Y, X)  – 채널 먼저
#       ├── coords  : (N, 3) int32  – (z, y, x) motor centres (voxel)
#       ├── spacing :  float        – Å / voxel
#       └── roi_start (optional)    – (z0, y0, x0) of current patch
#       └── rand_spatial_crop_*     – MONAI RandSpatialCrop meta
#  • dict 출력:
#       └── "label" (default) → (1, Z, Y, X) uint8
# ======================================================================
from __future__ import annotations
from typing import Sequence
import numpy as np
import monai.transforms as mt

__all__ = ["GenerateMotorMaskd"]


class GenerateMotorMaskd(mt.MapTransform):
    """Create a binary 3-D ball mask for each motor centre inside the patch."""

    def __init__(
        self,
        keys: Sequence[str] = ("image",),          # mandatory signature
        label_key: str = "label",
        radius_angstrom: float = 225.0,            # ≈ 22.5 nm
    ):
        super().__init__(keys)
        self.label_key = label_key
        self.r_ang = float(radius_angstrom)

    # ------------------------------------------------------------------
    @staticmethod
    def _origin_from_crop_meta(d: dict) -> np.ndarray:
        """Infer patch origin (z0,y0,x0)."""
        if "roi_start" in d:                       # dataset inserts this
            return np.asarray(d["roi_start"], dtype=int)

        for ck in ("rand_spatial_crop_center", "rand_spatial_crop_center_image"):
            if ck in d:
                center_full = np.asarray(d[ck], dtype=int)
                size_full = np.asarray(
                    d.get("rand_spatial_crop_size", d["image"].shape[1:]),  # strip channel
                    dtype=int,
                )
                return center_full[-3:] - size_full[-3:] // 2               # (z0,y0,x0)

        # full volume
        return np.zeros(3, dtype=int)

    # ------------------------------------------------------------------
    def __call__(self, data: dict):
        d = dict(data)                             # shallow copy

        if "coords" not in d or "spacing" not in d:
            raise KeyError("GenerateMotorMaskd needs 'coords' & 'spacing'.")

        origin = self._origin_from_crop_meta(d)    # (z0,y0,x0)
        spacing = float(d["spacing"])
        r = max(1, int(round(self.r_ang / spacing)))  # radius in voxels

        _, Z, Y, X = d["image"].shape              # channel first
        zz, yy, xx = np.ogrid[:Z, :Y, :X]
        mask = np.zeros((Z, Y, X), dtype=np.uint8)

        for zc, yc, xc in np.asarray(d["coords"], dtype=int):
            dz, dy, dx = zc - origin[0], yc - origin[1], xc - origin[2]
            if (
                -r <= dz < Z + r
                and -r <= dy < Y + r
                and -r <= dx < X + r
            ):
                mask[
                    (zz - dz) ** 2 + (yy - dy) ** 2 + (xx - dx) ** 2 <= r * r
                ] = 1

        d[self.label_key] = mask[None]             # add channel dim -> (1,Z,Y,X)
        return d
