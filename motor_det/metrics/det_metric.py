#  motor_det/metrics/det_metric.py
#  --------------------------------
"""
Fβ-score(β=2) 계산 모듈 – BYU Motor Challenge 전용
"""
from __future__ import annotations
import torch
from typing import Tuple

__all__ = ["match_tp_fp_fn", "fbeta_score"]


def match_tp_fp_fn(
    pred: torch.Tensor,              # (P,3) [Å]
    gt: torch.Tensor,                # (G,3) [Å]
    dist_thr: float = 1000.0,        # <= 1000 Å → TP
) -> Tuple[int, int, int]:
    """
    1 GT ↔ 1 Pred 매칭 (그리디) 후  TP/FP/FN 반환
    """
    if pred.numel() == 0:
        return 0, 0, int(gt.size(0))
    if gt.numel() == 0:
        return 0, int(pred.size(0)), 0

    dist = torch.cdist(pred, gt)           # (P,G)
    tp = 0
    gt_used = torch.zeros(gt.size(0), dtype=torch.bool, device=gt.device)

    # 가장 가까운 GT 부터 매칭
    for p in range(dist.size(0)):
        g = dist[p].argmin()
        if dist[p, g] <= dist_thr and not gt_used[g]:
            tp += 1
            gt_used[g] = True

    fp = pred.size(0) - tp
    fn = gt.size(0) - tp
    return tp, fp, fn


def fbeta_score(
    pred: torch.Tensor,
    gt: torch.Tensor,
    beta: float = 2.0,
    dist_thr: float = 1000.0,
) -> Tuple[float, float, float, int, int, int]:
    """
    단일 클래스 Fβ (=2) 스코어와 PR·카운트 반환
    """
    tp, fp, fn = match_tp_fp_fn(pred, gt, dist_thr)
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    beta2 = beta * beta
    fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-9)
    return fbeta, precision, recall, tp, fp, fn
