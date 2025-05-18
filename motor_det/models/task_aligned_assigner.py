from typing import Optional
import torch
from torch import Tensor, nn


def batch_pairwise_keypoints_iou(
    pred_keypoints: torch.Tensor,
    true_keypoints: torch.Tensor,
    true_sigmas: torch.Tensor,
) -> Tensor:
    """Calculate batched OKS (Object Keypoint Similarity) between two sets of keypoints."""
    centers1 = pred_keypoints[:, None, :, :]  # [B, M2, 1, 3]
    centers2 = true_keypoints[:, :, None, :]  # [B, 1, M1, 3]
    d = ((centers1 - centers2) ** 2).sum(dim=-1, keepdim=False)
    sigmas = true_sigmas.reshape(true_keypoints.size(0), true_keypoints.size(1), 1)
    e: Tensor = d / (2 * sigmas**2)
    return torch.exp(-e)


def compute_max_iou_anchor(ious: Tensor) -> Tensor:
    num_max_boxes = ious.shape[-2]
    max_iou_index = ious.argmax(dim=-2)
    is_max_iou: Tensor = torch.nn.functional.one_hot(max_iou_index, num_max_boxes).permute([0, 2, 1])
    return is_max_iou.type_as(ious)


def gather_topk_anchors(
    metrics: Tensor,
    topk: int,
    largest: bool = True,
    topk_mask: Optional[Tensor] = None,
    eps: float = 1e-9,
) -> Tensor:
    """Return mask of anchors falling in the top ``k`` metrics.

    ``torch.topk`` requires ``k <= num_anchors``.  Some datasets may yield fewer
    anchors than the configured ``topk``; clamp ``topk`` accordingly to avoid
    CUDA index errors.
    """
    num_anchors = metrics.shape[-1]
    topk = min(topk, num_anchors)
    topk_metrics, topk_idxs = torch.topk(metrics, topk, dim=-1, largest=largest)
    if topk_mask is None:
        topk_mask = (topk_metrics.max(dim=-1, keepdim=True).values > eps).type_as(metrics)
    is_in_topk = torch.nn.functional.one_hot(topk_idxs, num_anchors).sum(dim=-2).type_as(metrics)
    return is_in_topk * topk_mask


def check_points_inside_bboxes(
    anchor_points: Tensor,
    gt_centers: Tensor,
    gt_radius: Tensor,
    eps: float = 0.05,
) -> Tensor:
    """Return indicator if anchors fall inside the GT spheres.

    ``anchor_points`` may be 2‑D ``[N,3]`` or 3‑D ``[B,N,3]``.  When 2‑D, expand
    across the batch to match ``gt_centers``.
    """
    if anchor_points.ndim == 2:
        anchor_points = anchor_points.unsqueeze(0).expand(gt_centers.size(0), -1, -1)

    iou = batch_pairwise_keypoints_iou(anchor_points, gt_centers, gt_radius)
    return (iou > eps).type_as(gt_centers)


class TaskAlignedAssigner(nn.Module):
    def __init__(self, max_anchors_per_point, assigned_min_iou_for_anchor, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = max_anchors_per_point
        self.assigned_min_iou_for_anchor = assigned_min_iou_for_anchor
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pred_scores: Tensor,
        pred_centers: Tensor,
        anchor_points: Tensor,
        true_labels: Tensor,
        true_centers: Tensor,
        true_sigmas: Tensor,
        pad_gt_mask: Tensor,
        bg_index: int,
    ):
        assert pred_scores.ndim == pred_centers.ndim
        assert true_labels.ndim == true_centers.ndim and true_centers.ndim == 3

        batch_size, num_anchors, num_classes = pred_scores.shape
        _, num_max_boxes, _ = true_centers.shape

        if num_max_boxes == 0:
            assigned_labels = torch.full([batch_size, num_anchors], bg_index, dtype=torch.long, device=true_labels.device)
            assigned_points = torch.zeros([batch_size, num_anchors, 3], device=true_labels.device)
            assigned_scores = torch.zeros([batch_size, num_anchors, num_classes], device=true_labels.device)
            assigned_sigmas = torch.zeros([batch_size, num_anchors], device=true_labels.device)
            return assigned_labels, assigned_points, assigned_scores, assigned_sigmas

        ious = batch_pairwise_keypoints_iou(pred_centers, true_centers, true_sigmas)
        pred_scores = torch.permute(pred_scores, [0, 2, 1])
        batch_ind = torch.arange(end=batch_size, dtype=true_labels.dtype, device=true_labels.device).unsqueeze(-1)
        gt_labels_ind = torch.stack([batch_ind.tile([1, num_max_boxes]), true_labels.squeeze(-1)], dim=-1)
        bbox_cls_scores = pred_scores[gt_labels_ind[..., 0], gt_labels_ind[..., 1]]

        alignment_metrics = bbox_cls_scores.pow(self.alpha) * ious.pow(self.beta)
        is_in_gts = check_points_inside_bboxes(anchor_points, true_centers, true_sigmas, eps=self.assigned_min_iou_for_anchor)
        is_in_topk = gather_topk_anchors(alignment_metrics * is_in_gts, self.topk, topk_mask=pad_gt_mask)
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        mask_positive_sum = mask_positive.sum(dim=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).tile([1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = torch.where(mask_multiple_gts, is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(dim=-2)
        assigned_gt_index = mask_positive.argmax(dim=-2)

        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = torch.gather(true_labels.flatten(), index=assigned_gt_index.flatten(), dim=0)
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = torch.where(mask_positive_sum > 0, assigned_labels, torch.full_like(assigned_labels, bg_index))

        assigned_points = true_centers.reshape([-1, 3])[assigned_gt_index.flatten(), :]
        assigned_points = assigned_points.reshape([batch_size, num_anchors, 3])

        assigned_sigmas = true_sigmas.reshape([-1])[assigned_gt_index.flatten()]
        assigned_sigmas = assigned_sigmas.reshape([batch_size, num_anchors])

        assigned_scores = torch.nn.functional.one_hot(assigned_labels, num_classes + 1)
        ind = list(range(num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = torch.index_select(assigned_scores, index=torch.tensor(ind, device=assigned_scores.device, dtype=torch.long), dim=-1)

        alignment_metrics *= mask_positive
        max_metrics_per_instance = alignment_metrics.max(dim=-1, keepdim=True).values
        max_ious_per_instance = (ious * mask_positive).max(dim=-1, keepdim=True).values
        alignment_metrics = alignment_metrics / (max_metrics_per_instance + self.eps) * max_ious_per_instance
        alignment_metrics = alignment_metrics.max(dim=-2).values.unsqueeze(-1)
        assigned_scores = assigned_scores * alignment_metrics

        return assigned_labels, assigned_points, assigned_scores, assigned_sigmas
