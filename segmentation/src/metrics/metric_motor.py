# ======================================================================
#  metric_motor.py   – BYU Flagellar-motor FBeta(β=2) 평가 모듈
# ----------------------------------------------------------------------
#  • 결과는 대회 공식 score() 와 완전히 동일해야 한다
#  • KD-Tree 가속은 선택 사항(use_kdtree=True) – 기본 False
# ======================================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from typing import Union

try:
    from scipy.spatial import KDTree  # 가속용 (optional)
    _SCIPY_OK = True
except ImportError:  # scipy 가 없는 환경도 배려
    _SCIPY_OK = False


# ------------------------------------------------------------------ #
# 1. 공통 유틸                                                         #
# ------------------------------------------------------------------ #
COORD_COLS = ["Motor axis 0", "Motor axis 1", "Motor axis 2"]


def _has_motor(df: pd.DataFrame) -> np.ndarray:
    """좌표가 (-1,-1,-1)이 아니면 1, 아니면 0."""
    return (~(df[COORD_COLS] == -1).any(axis=1)).astype(int).values


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(((a - b) ** 2).sum(-1))


# ------------------------------------------------------------------ #
# 2. distance metric – 공식 로직과 동등                                 #
# ------------------------------------------------------------------ #
def _distance_pass(
    gt_xyz: np.ndarray,
    pred_xyz: np.ndarray,
    voxel_spacing: float,
    min_radius: float = 1000.0,
    thresh_ratio: float = 1.0,
) -> bool:
    """
    GT·예측 좌표 한 쌍이 TP 로 인정되는지 여부.
    """
    if (pred_xyz == -1).any():      # 예측이 ‘motor 없음’ 표시
        return False

    dist = _euclidean(gt_xyz, pred_xyz)
    thresh = (min_radius * thresh_ratio) / voxel_spacing  # Å→voxel
    return dist <= thresh


def _label_with_kdtree(
    gt_xyz: np.ndarray,
    pred_xyz: np.ndarray,
    voxel_spacing: np.ndarray,
    min_radius: float,
    thresh_ratio: float,
) -> np.ndarray:
    """
    KDTree 버전(많은 파티클에 유리).  
    이번 대회(≤1 motor)에는 속도 이득이 거의 없으므로 기본 off.
    """
    if not _SCIPY_OK:
        raise RuntimeError("scipy 가 설치되지 않아 KDTree 를 사용할 수 없습니다.")

    has_pred = ~(pred_xyz == -1).any(1)
    tree = KDTree(pred_xyz[has_pred])

    preds = has_pred.astype(int)  # 기본 라벨(0 또는 1)

    for i, (gt, vs) in enumerate(zip(gt_xyz, voxel_spacing)):
        if gt[0] == -1:           # GT 에 motor 없음
            continue
        radius = (min_radius * thresh_ratio) / vs
        idx = tree.query_ball_point(gt, r=radius)
        if len(idx) == 0:         # 거리에 들어오는 예측 없음 → FN
            preds[i] = 0
    return preds


# ------------------------------------------------------------------ #
# 3. 공개 score API                                                    #
# ------------------------------------------------------------------ #
def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    *,
    min_radius: float = 1000.0,
    beta: float = 2.0,
    use_kdtree: bool = False,
) -> float:
    """
    공식 FBeta 계산과 **동일한 결과**를 반환한다.
    """
    # ── ① 정렬·검증 ────────────────────────────────────────────────
    solution = solution.sort_values("tomo_id").reset_index(drop=True)
    submission = submission.sort_values("tomo_id").reset_index(drop=True)

    if not (solution["tomo_id"].values == submission["tomo_id"].values).all():
        raise ValueError("tomo_id 순서/구성이 ground-truth 와 다릅니다.")

    # ── ② ‘Has motor’ 라벨 부여 ────────────────────────────────────
    gt_has  = _has_motor(solution)
    sub_has = _has_motor(submission)

    preds = sub_has.copy()

    # ── ③ 거리-기반 라벨 수정 (둘 다 motor 존재할 때만) ───────────
    both_idx = np.where((gt_has == 1) & (sub_has == 1))[0]
    if both_idx.size:
        gt_xyz   = solution.loc[both_idx, COORD_COLS].values.astype(float)
        pred_xyz = submission.loc[both_idx, COORD_COLS].values.astype(float)
        voxel_sp = solution.loc[both_idx, "Voxel spacing"].values.astype(float)

        if use_kdtree and _SCIPY_OK and both_idx.size > 8:  # 파티클多에만 사용
            preds[both_idx] = _label_with_kdtree(
                gt_xyz, pred_xyz, voxel_sp,
                min_radius=min_radius, thresh_ratio=1.0,
            )
        else:  # 공식 코드와 같은 1:1 벡터화 방식
            dist_ok = (
                _euclidean(gt_xyz, pred_xyz)
                <= (min_radius / voxel_sp)
            )
            preds[both_idx] = dist_ok.astype(int)

    # ── ④ FBeta 계산 ──────────────────────────────────────────────
    return fbeta_score(gt_has, preds, beta=beta)


# ------------------------------------------------------------------ #
# 4. 파이프라인용 래퍼                                                 #
# ------------------------------------------------------------------ #
def calc_metric(
    cfg,
    pred_df: Union[pd.DataFrame, list],
    val_df: pd.DataFrame,
    pre: str = "val",
):
    """
    training/validation loop 에서 호출하기 위한 thin wrapper.

    Returns
    -------
    dict  e.g. {"score": 0.842}
    """
    if isinstance(pred_df, list):   # (val_data, gt_df) 형태 대비
        pred_df = pred_df[0]

    try:
        sc = score(
            solution   = val_df,
            submission = pred_df,
            min_radius = getattr(cfg, "min_radius", 1000.0),
            beta       = getattr(cfg, "fbeta", 2.0),
            use_kdtree = getattr(cfg, "metric_use_kdtree", False),
        )
    except Exception as e:
        # metric 계산 실패 시 0 반환 & 에러 로그
        print(f"[metric] {pre} metric 계산 실패: {e}")
        sc = 0.0

    return {"score": sc}
