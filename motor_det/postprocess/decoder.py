#  motor_det/postprocess/decoder.py
#  --------------------------------
"""
Heat-map + Offset → Å 좌표 디코더 (+ 간단 3-D NMS)
"""
from __future__ import annotations
import torch
from functools import lru_cache
from typing import List, Tuple

__all__ = [
    "decode_detections",
    "decode_with_nms",
    "vectorized_nms",
]


@lru_cache(maxsize=None)
def _cached_anchor_grid(shape: Tuple[int, int, int], stride: int, device: torch.device) -> torch.Tensor:
    D, H, W = shape
    z, y, x = torch.meshgrid(
        torch.arange(D, device=device),
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    anchors = torch.stack([x, y, z], dim=0).float().add_(0.5).mul_(stride)
    return anchors


def anchors_for_offsets_feature_map(shape: Tuple[int, int, int], stride: int, device=None) -> torch.Tensor:
    """(D,H,W) feature map → anchor 좌표(Å) tensor (3,D,H,W)"""
    device = torch.device("cpu") if device is None else torch.device(device)
    return _cached_anchor_grid(tuple(shape), stride, device)


def decode_detections(
    logits: torch.Tensor,         # (B,1,D,H,W)
    offsets: torch.Tensor,        # (B,3,D,H,W)
    stride: int = 2,
    prob_thr: float = 0.5,
) -> List[torch.Tensor]:
    """
    배치-별 예측 좌표(Å) 리스트 반환
    """
    B = logits.size(0)
    anchors = anchors_for_offsets_feature_map(logits.shape[-3:], stride, logits.device)  # (3,D,H,W)
    coords = anchors + offsets                       # (B,3,D,H,W)
    prob   = logits.sigmoid()                       # (B,1,D,H,W)

    centers_out: List[torch.Tensor] = []
    for b in range(B):
        mask = prob[b, 0] >= prob_thr               # (D,H,W)
        if mask.any():
            pts = coords[b, :, mask].T              # (N,3)
            centers_out.append(pts)
        else:
            centers_out.append(torch.empty((0, 3), device=logits.device))
    return centers_out


def greedy_nms(centers: torch.Tensor, sigma: float = 60.0, iou_thr: float = 0.25) -> torch.Tensor:
    """
    IoU(=exp(-d²/2σ²)) 기반 그리디 NMS
    """
    if centers.size(0) == 0:
        return centers
    keep: List[int] = []
    suppressed = torch.zeros(centers.size(0), dtype=torch.bool, device=centers.device)
    for i in range(centers.size(0)):
        if suppressed[i]:
            continue
        keep.append(i)
        d2 = ((centers[i] - centers) ** 2).sum(dim=1)
        iou = torch.exp(-d2 / (2 * sigma * sigma))
        suppressed |= iou > iou_thr
    return centers[torch.tensor(keep, device=centers.device)]


def vectorized_nms(centers: torch.Tensor, sigma: float = 60.0, iou_thr: float = 0.25) -> torch.Tensor:
    """Vectorised NMS to speed up inference."""
    if centers.size(0) == 0:
        return centers
    order = torch.arange(centers.size(0), device=centers.device)
    keep = torch.ones_like(order, dtype=torch.bool)
    d2 = ((centers.unsqueeze(0) - centers.unsqueeze(1)) ** 2).sum(dim=-1)
    iou = torch.exp(-d2 / (2 * sigma * sigma))
    for i in order:
        if not keep[i]:
            continue
        keep &= iou[i] <= iou_thr
        keep[i] = True
    return centers[keep]


def decode_with_nms(
    logits: torch.Tensor, offsets: torch.Tensor, stride: int = 2,
    prob_thr: float = 0.5, sigma: float = 60.0, iou_thr: float = 0.25
) -> List[torch.Tensor]:
    """
    Heatmap+Offset → NMS 후 Å 좌표 리스트
    """
    centers_batch = decode_detections(logits, offsets, stride, prob_thr)
    return [vectorized_nms(c, sigma, iou_thr) for c in centers_batch]
