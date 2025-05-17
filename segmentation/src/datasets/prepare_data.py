# ======================================================================
#  src/datasets/prepare_data.py
# ----------------------------------------------------------------------
#  JPEG stack  →  3-D Zarr converter   (train / test 共用)
# ----------------------------------------------------------------------
#  Improvements over the baseline:
#   1. Robust XY-flip detection   2. Optional voxel-spacing resample
#   3. Intensity percentile-norm  4. Detailed logging / CLI
# ======================================================================
from __future__ import annotations
import argparse, signal, sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import imageio.v2 as imageio              # suppress v3 warning
import zarr, scipy.ndimage                # for zoom
from tqdm import tqdm


# ------------------------------------------------------------------ #
# 0. helper : graceful Ctrl-C                                           
# ------------------------------------------------------------------ #
def _sigint_handler(sig, frame):
    print("\nInterrupted ‒ exiting early.")
    sys.exit(130)
signal.signal(signal.SIGINT, _sigint_handler)


# ------------------------------------------------------------------ #
# 1. load expected (Z,Y,X) shapes from train_labels.csv                 
# ------------------------------------------------------------------ #
def load_shape_dict(label_csv: Path) -> Dict[str, Tuple[int, int, int]]:
    if not label_csv.exists():
        return {}
    df = (np.genfromtxt(label_csv, delimiter=",", names=True, dtype=None,
                        encoding=None))
    shape_dict = {}
    for row in df:
        shape_dict[row["tomo_id"]] = (
            int(row["Array_shape_axis_0"]),
            int(row["Array_shape_axis_1"]),
            int(row["Array_shape_axis_2"]),
        )
    return shape_dict


# ------------------------------------------------------------------ #
# 2. percentile normalisation (0–255)                                   
# ------------------------------------------------------------------ #
def percentile_norm(img: np.ndarray,
                    p_low: float = .5,
                    p_high: float = 99.5,
                    dtype=np.uint8) -> np.ndarray:
    lo, hi = np.percentile(img, (p_low, p_high))
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-7)
    img = (img * np.iinfo(dtype).max).astype(dtype)
    return img


# ------------------------------------------------------------------ #
# 3. resample to target voxel spacing                                   
# ------------------------------------------------------------------ #
def resample_isotropic(vol: np.ndarray,
                       src_spacing: float,
                       tgt_spacing: float,
                       order: int = 3) -> np.ndarray:
    if np.isclose(src_spacing, tgt_spacing, rtol=1e-2):
        return vol
    zoom = src_spacing / tgt_spacing          # >1 upsample  <1 downsample
    vol = scipy.ndimage.zoom(vol,
                             zoom=(zoom, zoom, zoom),
                             order=order,
                             mode="reflect")
    return vol


# ------------------------------------------------------------------ #
# 4. decide whether XY swap is needed
#        * 정사각형 볼륨(axis-1 == axis-2)은 strict 비교에서 제외 *
# ------------------------------------------------------------------ #
def need_flip_xy(h: int, w: int,
                 exp_shape: Optional[Tuple[int, int, int]]) -> bool:
    """
    True  → (H,W) 가 (X,Y) 순서·뒤집힘이므로 XY 전치 필요
    False → 그대로 저장
    """
    # ── ① 레이블 CSV의 shape 를 신뢰할 수 있을 때 ────────────────
    if exp_shape is not None:
        z, y, x = exp_shape
        if y != x:                       # ★ 정사각형이면 strict 체크 skip
            return (h, w) == (x, y)
        # 정사각형이면 정보 부족 → 휴리스틱으로 넘어감

    # ── ② 휴리스틱 (레이블 shape 없거나 정사각형) ─────────────────
    #   aspect ratio 가 1.4 이상이면서  h<w  → 뒤집힌 경우가 많다
    ar = h / w if w else 1.0
    if ar < 0.7 or ar > 1.4:
        return h < w

    #   그 외엔 뒤집지 않는다
    return False


# ------------------------------------------------------------------ #
# 5. core conversion routine                                            
# ------------------------------------------------------------------ #
def build_zarr(jpeg_dir: Path,
               out_path: Path,
               exp_shape: Optional[Tuple[int, int, int]],
               *,
               step: int,
               tgt_dtype: np.dtype,
               tgt_spacing: Optional[float],
               do_norm: bool):
    jpg_files = sorted([p for p in jpeg_dir.iterdir()
                        if p.suffix.lower() == ".jpg"])[::step]
    if not jpg_files:
        print(f"[WARN] no JPEG in {jpeg_dir}")
        return

    # probe first slice
    img0 = imageio.imread(jpg_files[0])
    H, W = img0.shape[:2]

    flip_xy = need_flip_xy(H, W, exp_shape)
    if flip_xy:
        H, W = W, H

    Z = len(jpg_files)
    zroot = zarr.open(out_path, mode="w",
                      shape=(Z, H, W),
                      chunks=(1, H, W),
                      dtype=tgt_dtype,
                      compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=1))

    # write slices
    for zi, fp in enumerate(tqdm(jpg_files,
                                 desc=f"[{jpeg_dir.name}]",
                                 leave=False)):
        img = imageio.imread(fp)
        if flip_xy:
            img = img.T
        if do_norm:
            img = percentile_norm(img, dtype=tgt_dtype)
        else:
            img = img.astype(tgt_dtype)
        zroot[zi] = img

    # attrs
    zroot.attrs["flip_xy"] = bool(flip_xy)
    if "voxel_spacing" in img0.meta if hasattr(img0, "meta") else {}:
        src_spacing = float(img0.meta["voxel_spacing"])
    else:
        src_spacing = None
    if src_spacing is not None:
        zroot.attrs["voxel_spacing"] = src_spacing

    # optional resample
    if tgt_spacing and src_spacing:
        arr = zroot[:]                 # load to memory once
        arr = resample_isotropic(arr, src_spacing, tgt_spacing)
        # overwrite
        del zroot[:]                   # free old
        zroot.resize(arr.shape)
        zroot[:] = arr
        zroot.attrs["voxel_spacing"] = tgt_spacing
        zroot.attrs["was_resampled"] = True
    else:
        zroot.attrs["was_resampled"] = False

    print(f"✓ {jpeg_dir.name}  →  {zroot.shape} "
          f"{'(flipXY)' if flip_xy else ''}"
          f"{'(resampled)' if zroot.attrs['was_resampled'] else ''}")


# ------------------------------------------------------------------ #
# 6. CLI entry                                                         
# ------------------------------------------------------------------ #
def parse_args():
    ap = argparse.ArgumentParser(
        description="JPEG stack → Zarr converter (BYU motor dataset)")
    ap.add_argument("--raw_root", required=True,
                    help="root containing train/ test/ directories")
    ap.add_argument("--out_root", required=True,
                    help="destination root for .zarr")
    ap.add_argument("--step", type=int, default=1,
                    help="undersample factor along Z (default 1)")
    ap.add_argument("--dtype", default="uint8",
                    choices=["uint8", "uint16", "float16", "float32"],
                    help="output voxel dtype")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing .zarr")
    ap.add_argument("--tgt_spacing", type=float, default=None,
                    help="if given, resample to this Å/voxel")
    ap.add_argument("--no_norm", action="store_true",
                    help="skip intensity percentile normalisation")
    return ap.parse_args()


def main():
    args = parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # expected shapes from train_labels (if present)
    shape_dict = load_shape_dict(raw_root / "train_labels.csv")

    for split in ["train", "test"]:
        src_dir = raw_root / split
        if not src_dir.exists():
            continue
        dst_dir = out_root / split
        dst_dir.mkdir(parents=True, exist_ok=True)

        for tomo_dir in sorted(src_dir.iterdir()):
            if not tomo_dir.is_dir():
                continue
            out_fp = dst_dir / f"{tomo_dir.name}.zarr"
            if out_fp.exists() and not args.overwrite:
                continue
            try:
                build_zarr(jpeg_dir=tomo_dir,
                           out_path=out_fp,
                           exp_shape=shape_dict.get(tomo_dir.name),
                           step=args.step,
                           tgt_dtype=np.dtype(args.dtype),
                           tgt_spacing=args.tgt_spacing,
                           do_norm=not args.no_norm)
            except Exception as e:
                print(f"[ERROR] {tomo_dir.name}: {e}")

    print("All done.")


if __name__ == "__main__":
    main()
