# ======================================================================
#  src/models/losses.py
# ----------------------------------------------------------------------
#  Common loss functions / utilities  (3-D tomogram segmentation)
# ======================================================================
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

# ----------------------------------------------------------------------
# 기존 모듈 내에 있던 MixUp --> 별도 파일로 이관 (호환 위해 import)
# ----------------------------------------------------------------------
from models.mixup import Mixup        # noqa: F401  (re-export)


# ----------------------------- utils -------------------------------- #
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def human_format(num: int | float) -> str:
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return f"{num:.1f}{' KMBT'[magnitude]}"


# ---------------------- one-hot helper (CE용) ------------------------ #
def to_ce_target(y: torch.Tensor) -> torch.Tensor:
    """
    Sparse (C, …) mask → dense (C+1, …) soft-target for Cross-Entropy.
    y 에는 배경 채널이 포함되지 않은 상태로 가정.
    """
    bg = 1.0 - y.sum(1, keepdim=True).clamp(0, 1)
    tgt = torch.cat([y, bg], 1)          # (C+1, …)
    tgt /= tgt.sum(1, keepdim=True)
    return tgt


# ------------------- Dense Cross-Entropy with weights ---------------- #
class DenseCrossEntropy(nn.Module):
    """
    CE(logits, target_soft).  
    target_soft  : one-hot 또는 soft-label (C+1, …)
    class_weights: (C+1,)  Tensor (background 포함 가능)
    """

    def __init__(self, class_weights: torch.Tensor | None = None):
        super().__init__()
        if class_weights is not None:
            self.register_buffer("w", class_weights.float())
        else:
            self.w = None

    def forward(self,
                logits: torch.Tensor,
                target_soft: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logp = F.log_softmax(logits.float(), dim=1)
        loss = -(target_soft * logp)

        # spatial mean  (B, C+1)
        loss = loss.mean(dim=tuple(range(2, loss.ndim)))

        # class weighting
        if self.w is not None:
            loss = loss * self.w.view(1, -1)

        loss_per_class = loss.mean(0)        # (C+1,)
        total = loss_per_class.sum()
        return total, loss_per_class.detach()


# -------------------------- Dice (soft) Loss ------------------------- #
class DiceLoss(nn.Module):
    """
    Soft Dice loss  (1 - Dice).  
    Works for logits or probabilities -> ``apply_sigmoid``/``apply_softmax``.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 apply_sigmoid: bool = True,
                 apply_softmax: bool = False):
        super().__init__()
        self.eps = eps
        self.apply_sigmoid = apply_sigmoid
        self.apply_softmax = apply_softmax

    def forward(self,
                logits_or_prob: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        logits_or_prob : (B, C, …)   – C==1 => binary
        target         : same shape (soft mask or {0,1})
        """
        if self.apply_sigmoid and logits_or_prob.size(1) == 1:
            prob = torch.sigmoid(logits_or_prob)
        elif self.apply_softmax and logits_or_prob.size(1) > 1:
            prob = F.softmax(logits_or_prob, dim=1)
        else:
            prob = logits_or_prob

        prob = prob.contiguous().view(prob.size(0), prob.size(1), -1)
        tgt  = target.contiguous().view(target.size(0), target.size(1), -1)

        intersect = (prob * tgt).sum(-1)
        denom     = prob.sum(-1) + tgt.sum(-1)

        dice = (2. * intersect + self.eps) / (denom + self.eps)
        loss = 1. - dice
        return loss.mean()                 # over B & C


# --------------------------- Focal Loss ------------------------------ #
class FocalLoss(nn.Module):
    """
    Focal Loss (α-balanced, γ-focused)  
    • binary  → sigmoid  /   multi-class → softmax
    """

    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self,
                logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        target : one-hot (same shape as logits after softmax/sigmoid)
                 또는 binary {0,1} for C=1
        """
        if logits.size(1) == 1:                           # binary
            prob = torch.sigmoid(logits)
            tgt  = target
            ce_loss = F.binary_cross_entropy_with_logits(
                logits, tgt, reduction="none")
        else:                                             # multi-class
            prob = F.softmax(logits, 1)
            tgt  = target
            ce_loss = -(tgt * torch.log_softmax(logits, 1)).sum(1, keepdim=True)

        pt = (prob * tgt + 1e-8).sum(1, keepdim=True)     # prob of true class
        focal = self.alpha * (1. - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


# ------------------------ Focal + Dice hybrid ------------------------ #
class FocalDiceLoss(nn.Module):
    """
    λ * Focal + (1-λ) * Dice  (default λ=0.5  → 동일 비중)
    """

    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 lambda_focal: float = 0.5):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")
        self.dice  = DiceLoss(apply_sigmoid=False, apply_softmax=False)
        self.lambda_f = lambda_focal

    def forward(self,
                logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(logits, target)
        dice_loss  = self.dice(logits, target)
        return self.lambda_f * focal_loss + (1. - self.lambda_f) * dice_loss
