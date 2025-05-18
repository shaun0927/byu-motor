"""
motor_det.loss.losses
=====================
간결한 1-클래스 BYU 모터 탐지 손실 함수 (Focal + IoU)
------------------------------------
Batch dict 필수 key
    "cls"    : (B,1,D',H',W')  0/1 binary GT heat-map
    "offset" : (B,3,D',H',W')  Δx,Δy,Δz  (centre voxel에서만 유효)

Model output dict
    "cls", "offset" 같은 shape
"""
from __future__ import annotations
from typing import Dict, Tuple

from motor_det.postprocess.decoder import anchors_for_offsets_feature_map

import torch
import torch.nn.functional as F
from torch import Tensor


def keypoint_similarity(p1: Tensor, p2: Tensor, sigma: float) -> Tensor:
    """Gaussian keypoint similarity used as IoU approximation."""
    d2 = ((p1 - p2) ** 2).sum(dim=-1)
    return torch.exp(-d2 / (2 * sigma * sigma))


def varifocal_loss(pred_logits: Tensor, gt_score: Tensor, label: Tensor, *, alpha: float = 0.75, gamma: float = 2.0) -> Tensor:
    """Varifocal loss as used in many anchor-free detectors."""
    pred_score = pred_logits.sigmoid()
    weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
    bce = F.binary_cross_entropy_with_logits(pred_logits, gt_score, reduction="none")
    return (weight * bce).mean()


def focal_loss(pred_logits: Tensor, label: Tensor, *, alpha: float = 0.25, gamma: float = 2.0) -> Tensor:
    """Standard binary focal loss."""
    prob = pred_logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(pred_logits, label, reduction="none")
    p_t = prob * label + (1 - prob) * (1 - label)
    alpha_factor = alpha * label + (1 - alpha) * (1 - label)
    modulating = (1 - p_t).pow(gamma)
    return (alpha_factor * modulating * ce).mean()


def motor_detection_loss(
    pred: Dict[str, Tensor],
    batch: Dict[str, Tensor],
    lambda_offset: float = 1.0,
    *,
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
    pos_weight_clip: float = 5.0,  # kept for API compatibility
    pos_weight: Tensor | None = None,
    sigma: float = 60.0,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Compute object detection loss using focal classification and IoU based
    regression.

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

    # build anchor grid to convert offsets → absolute centres
    _, _, D, H, W = pred_cls.shape
    anchors = anchors_for_offsets_feature_map((D, H, W), stride=2, device=pred_cls.device).unsqueeze(0)
    pred_center = anchors + pred_off
    gt_center = anchors + gt_off

    # IoU-like similarity between predicted and GT centres
    diff2 = (pred_center - gt_center).pow(2).sum(dim=1, keepdim=True)
    iou_map = torch.exp(-diff2 / (2 * sigma * sigma))

    # ─── 2. Classification: Focal ───────────────────────────────────────
    cls_loss = focal_loss(
        pred_cls.float(),
        gt_cls.float(),
        alpha=focal_alpha,
        gamma=focal_gamma,
    )

    # ─── 3. Regression IoU + L1 (positive 위치에서만) ────────────
    pos_mask: Tensor = gt_cls.bool()
    if pos_mask.any():
        pos_mask_3 = pos_mask.expand_as(pred_center)
        iou_loss = (1 - iou_map[pos_mask]).mean()
        l1 = F.smooth_l1_loss(
            pred_center[pos_mask_3].float(),
            gt_center[pos_mask_3].float(),
            reduction="mean",
        )
    else:
        iou_loss = torch.tensor(0.0, device=pred_cls.device)
        l1 = torch.tensor(0.0, device=pred_cls.device)

    reg_loss = iou_loss + l1

    # ─── 4. 총 손실 및 로그 ──────────────────────────────────
    loss = cls_loss + lambda_offset * reg_loss
    logs = {
        "loss":       float(loss),
        "loss_cls":   float(cls_loss),
        "loss_off":   float(reg_loss),
        "num_pos":    float(pos_mask.sum().item()),
    }
    return loss, logs
