"""
motor_det.loss.losses
=====================
간결한 1-클래스 BYU 모터 탐지 손실 함수
------------------------------------
Batch dict 필수 key
    "cls"    : (B,1,D',H',W')  0/1 binary GT heat-map
    "offset" : (B,3,D',H',W')  Δx,Δy,Δz  (centre voxel에서만 유효)

Model output dict
    "cls", "offset" 같은 shape
"""
from __future__ import annotations
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def motor_detection_loss(
    pred: Dict[str, Tensor],
    batch: Dict[str, Tensor],
    lambda_offset: float = 1.0,
    *,
    focal_gamma: float = 2.0,
    pos_weight_clip: float = 5.0,
    pos_weight: Tensor | None = None,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Returns
    -------
    loss  : scalar tensor
    logs  : dict(str -> float)  (for Lightning self.log_dict)
    pos_weight : optional positive class weight for BCEWithLogits

    Parameters
    ----------
    focal_gamma : exponent for focal loss.  Set ``None`` to disable.
    pos_weight_clip : maximum allowed positive weight to prevent large imbalance.
    """
    # ─── 1. 분리 ───────────────────────────────────────────────
    pred_cls:  Tensor = pred["cls"]    # (B,1,D',H',W')
    pred_off: Tensor = pred["offset"]  # (B,3,D',H',W')

    gt_cls:   Tensor = batch["cls"]    # same shape
    gt_off:   Tensor = batch["offset"] # same shape

    # ─── 2. Classification BCEWithLogits ──────────────────────────────
    with torch.cuda.amp.autocast(enabled=False):        # <─ AMP off
        if pos_weight is None:
            pos = gt_cls.sum()
            neg = gt_cls.numel() - pos
            w = (neg / (pos + 1e-9)) if pos > 0 else 1.0
            w = float(min(w, pos_weight_clip))
            pos_weight = torch.tensor(w, device=pred_cls.device)

        bce_raw = F.binary_cross_entropy_with_logits(
            pred_cls.float(),       # logits (32-bit)
            gt_cls.float(),
            weight=None,
            pos_weight=pos_weight,
            reduction="none",
        )

        if focal_gamma is not None and focal_gamma > 0:
            prob = torch.sigmoid(pred_cls.float())
            pt = prob * gt_cls + (1.0 - prob) * (1.0 - gt_cls)
            bce_raw = bce_raw * ((1.0 - pt) ** focal_gamma)

        bce = bce_raw.mean()

    # ─── 3. Offset L1 (positive 위치에서만) ──────────────────
    pos_mask: Tensor = gt_cls.bool()                 # (B,1,D',H',W')
    if pos_mask.any():
        # expand mask to 3 channels
        pos_mask_3 = pos_mask.expand_as(pred_off)    # same shape as offset
        l1 = F.l1_loss(
            pred_off[pos_mask_3].float(),
            gt_off[pos_mask_3].float(),
            reduction="mean",
        )
    else:
        l1 = torch.tensor(0.0, device=pred_off.device)

    # ─── 4. 총 손실 및 로그 ──────────────────────────────────
    loss = bce + lambda_offset * l1
    logs = {
        "loss":       float(loss),
        "loss_cls":   float(bce),
        "loss_off":   float(l1),
        "num_pos":    float(pos_mask.sum().item()),
    }
    return loss, logs
