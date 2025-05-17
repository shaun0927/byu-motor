"""
Utilities for BYU Motor dataset
===============================
- voxel_spacing_map : train_labels.csv → {tomo_id: Å/voxel}
- read_train_centers: dataframe (tomo_id, z, y, x, fold)
"""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import pandas as pd

# test 세트 기본값 (요청 15 Å/voxel)
DEFAULT_TEST_SPACING = 15.0

# ----------------------------- voxel spacing ---------------------------------
@lru_cache(maxsize=1)
def voxel_spacing_map(root: str | Path) -> Dict[str, float]:
    """
    Returns dict[tomo_id -> voxel_spacing(Å)] for BYU train set.
    Test tomograms( spacing 미공개 )는 기본값 15 Å 로 채움.
    """
    root = Path(root)
    csv_path = root / "raw" / "train_labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if "Voxel spacing" not in df.columns:
        raise ValueError("'Voxel spacing' column not found in train_labels.csv")

    return dict(zip(df["tomo_id"].astype(str), df["Voxel spacing"].astype(float)))


# ----------------------------- train centres ---------------------------------
def read_train_centers(root: str | Path) -> pd.DataFrame:
    """
    Returns DataFrame with BYU columns kept 그대로:
      ['tomo_id', 'Motor axis 0', 'Motor axis 1', 'Motor axis 2', 'fold']
    (fold 컬럼이 없다면 -1 로 채움)
    Negative 'Motor axis 0' ⇒ 모터 없음.
    """
    root = Path(root)
    csv_path = root / "raw" / "train_labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    # fold 컬럼이 없을 수 있으므로 대비
    if "fold" not in df.columns:
        df["fold"] = -1

    keep = ["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2", "fold"]
    df = df[keep].copy()
    df["tomo_id"] = df["tomo_id"].astype(str)
    return df


# ----------------------------- test IDs --------------------------------------
def read_test_ids(root: str | Path) -> List[str]:
    """
    Returns list of tomo_id strings found in processed/zarr/test directory.
    """
    root = Path(root)
    test_dir = root / "processed" / "zarr" / "test"
    if not test_dir.exists():
        return []
    return [p.stem for p in test_dir.glob("*.zarr")]
