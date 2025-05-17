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
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Returns
    -------
    loss  : scalar tensor
    logs  : dict(str -> float)  (for Lightning self.log_dict)
    """
    # ─── 1. 분리 ───────────────────────────────────────────────
    pred_cls:  Tensor = pred["cls"]    # (B,1,D',H',W')
    pred_off: Tensor = pred["offset"]  # (B,3,D',H',W')

    gt_cls:   Tensor = batch["cls"]    # same shape
    gt_off:   Tensor = batch["offset"] # same shape

    # ─── 2. Classification BCE  ──────────────────────────────
    with torch.cuda.amp.autocast(enabled=False):        # <─ AMP off
        bce = F.binary_cross_entropy(
            pred_cls.float(),       # 32-bit
            gt_cls.float(),
            reduction="mean",
        )

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
