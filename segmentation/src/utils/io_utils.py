"""
io_utils.py
────────────────────────────────────────
• CSV / Parquet 읽기 (read_dataframe)
• JPEG stack ↔ Zarr  I/O 편의 함수(load_zarr / save_zarr)
• 기타 파일/디렉터리 유틸
"""

from __future__ import annotations
import shutil, json, gzip
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import imageio.v2 as imageio
import zarr


# ------------------------------------------------------------------ #
# 1. dataframe loader
# ------------------------------------------------------------------ #
def read_dataframe(src: str | Path,
                   infer_index: bool = False) -> pd.DataFrame:
    """
    CSV / Parquet 모두 지원.
    """
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(src)
    if src.suffix.lower() == ".parquet":
        df = pd.read_parquet(src, engine="fastparquet")
    else:
        df = pd.read_csv(src)
    if infer_index and "index" in df.columns:
        df = df.set_index("index")
    return df


# ------------------------------------------------------------------ #
# 2. Zarr helpers
# ------------------------------------------------------------------ #
def load_zarr(zarr_path: str | Path, as_numpy: bool = True) -> np.ndarray | zarr.Array:
    """
    zarr array 를 메모리로 읽어 numpy 로 반환.
    큰 볼륨이라면 as_numpy=False 후 lazy-loading 으로 사용 가능.
    """
    z = zarr.open(zarr_path, mode="r")
    return z[:] if as_numpy else z


def save_zarr(array: np.ndarray,
              out_path: str | Path,
              chunks: Tuple[int, ...] | None = None,
              compressor: zarr.Compressor | None = None,
              **attrs):
    """
    numpy array → zarr 저장. attrs 딕셔너리는 zroot.attrs 로 추가.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if chunks is None:
        # default : slice-wise chunk (z, H, W) → (1, H, W)
        chunks = (1, *array.shape[1:])

    zroot = zarr.open(out_path, mode="w",
                      shape=array.shape,
                      dtype=array.dtype,
                      chunks=chunks,
                      compressor=compressor or zarr.Blosc(
                          cname="zstd", clevel=3, shuffle=1))
    zroot[:] = array
    for k, v in attrs.items():
        zroot.attrs[k] = v


# ------------------------------------------------------------------ #
# 3. image stack utilities (JPEG ↔ numpy)
# ------------------------------------------------------------------ #
def read_jpeg_stack(jpeg_dir: Path,
                    step: int = 1,
                    dtype=np.uint8) -> np.ndarray:
    """
    jpeg 디렉터리를 (Z, Y, X) numpy volume 으로 읽어들임.
    """
    files = sorted([p for p in jpeg_dir.iterdir()
                    if p.suffix.lower() == ".jpg"])[::step]
    if not files:
        raise RuntimeError(f"No JPEG found in {jpeg_dir}")
    slices = [imageio.imread(f).astype(dtype) for f in files]
    return np.stack(slices, axis=0)


def save_jpeg_stack(volume: np.ndarray,
                    out_dir: Path,
                    ext: str = ".jpg"):
    """
    3-D volume → slice-wise JPEG 저장 (debug 용도)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for z, slice_ in enumerate(volume):
        fp = out_dir / f"{z:04d}{ext}"
        imageio.imwrite(fp, slice_)


# ------------------------------------------------------------------ #
# 4. json & gzip helpers (옵션)
# ------------------------------------------------------------------ #
def read_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Dict[str, Any], path: str | Path, indent: int = 2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def gzip_compress(src: str | Path, dst: str | Path | None = None):
    src, dst = Path(src), Path(dst or f"{src}.gz")
    with open(src, "rb") as f_in, gzip.open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
