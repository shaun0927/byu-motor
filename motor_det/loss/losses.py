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

    # ─── 2. Classification: Varifocal ──────────────────────────────────
    gt_score = iou_map * gt_cls
    cls_loss = varifocal_loss(
        pred_cls.float(),
        gt_score.float(),
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


def task_aligned_detection_loss(
    logits_list: list[Tensor],
    offsets_list: list[Tensor],
    strides: list[int],
    centers_list: list[Tensor],
    spacings: Tensor,
    assigner,
    sigma_voxel: float = 5.0,
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> tuple[Tensor, dict[str, float]]:
    """Compute loss for multi-scale FCOS head using :class:`TaskAlignedAssigner`."""

    device = logits_list[0].device
    batch_size = spacings.size(0)

    # --- prepare ground truth tensors ------------------------------------
    n_max = max(c.size(0) for c in centers_list)
    true_centers = torch.zeros(batch_size, n_max, 3, device=device)
    true_labels = torch.ones(batch_size, n_max, 1, dtype=torch.long, device=device)
    true_sigmas = torch.full((batch_size, n_max), sigma_voxel, device=device)
    pad_mask = torch.zeros(batch_size, n_max, 1, device=device)

    for b, (c, sp) in enumerate(zip(centers_list, spacings)):
        if c.numel() == 0:
            continue
        n = c.size(0)
        true_centers[b, :n] = c / sp
        pad_mask[b, :n, 0] = 1.0

    # --- flatten predictions --------------------------------------------
    pred_scores = []
    pred_logits = []
    pred_centers = []
    anchors_all = []
    for lg, off, stride in zip(logits_list, offsets_list, strides):
        B, _, D, H, W = lg.shape
        anchors = anchors_for_offsets_feature_map((D, H, W), stride, device).view(3, -1).permute(1, 0)
        centers = off.view(B, 3, -1).permute(0, 2, 1) + anchors.unsqueeze(0)
        pred_centers.append(centers)
        pred_scores.append(lg.sigmoid().view(B, 1, -1).permute(0, 2, 1))
        pred_logits.append(lg.view(B, 1, -1).permute(0, 2, 1))
        anchors_all.append(anchors)

    pred_scores = torch.cat(pred_scores, dim=1)
    pred_logits = torch.cat(pred_logits, dim=1)
    pred_centers = torch.cat(pred_centers, dim=1)
    anchors_cat = torch.cat(anchors_all, dim=0)

    lbl, tgt_ctr, tgt_scr, tgt_sigma = assigner(
        pred_scores,
        pred_centers,
        anchors_cat,
        true_labels,
        true_centers,
        true_sigmas,
        pad_mask,
        bg_index=0,
    )

    label_mask = (lbl > 0).unsqueeze(-1)
    cls_loss = varifocal_loss(pred_logits.squeeze(-1), tgt_scr.squeeze(-1), label_mask.float(), alpha=alpha, gamma=gamma)

    if label_mask.any():
        diff2 = (pred_centers - tgt_ctr).pow(2).sum(dim=-1)
        iou = torch.exp(-diff2 / (2 * tgt_sigma**2))
        iou_loss = (1 - iou[label_mask.squeeze(-1)]).mean()
        l1 = F.smooth_l1_loss(
            pred_centers[label_mask.squeeze(-1)],
            tgt_ctr[label_mask.squeeze(-1)],
            reduction="mean",
        )
        reg_loss = iou_loss + l1
    else:
        reg_loss = torch.tensor(0.0, device=device)

    loss = cls_loss + reg_loss
    logs = {
        "loss": float(loss),
        "loss_cls": float(cls_loss),
        "loss_off": float(reg_loss),
        "num_pos": float(label_mask.sum().item()),
    }
    return loss, logs
