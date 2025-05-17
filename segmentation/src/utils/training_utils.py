"""
training_utils.py
────────────────────────────────────────
• reproducibility helper (set_seed)
• metric / loss 평균 계산 (AverageMeter)
• AMP & cuDNN 환경 설정 (setup_amp_env)
• checkpoint save / load
• grad·weight norm 계산
"""

from __future__ import annotations
import os, random, logging, math, shutil, time
from pathlib import Path
from types   import SimpleNamespace
from typing  import Iterable, Dict, Any

import numpy as np
import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler


# ------------------------------------------------------------------ #
# 1. reproducibility + cuDNN/TF32 설정
# ------------------------------------------------------------------ #
def set_seed(seed: int = 42, deterministic: bool = False):
    """시드 고정 & cuDNN 설정."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark     = not deterministic

    # FP32 matmul 정밀도(AMP‧TF32) – Torch 2.0+
    #   • high  : TF32 off
    #   • medium: TF32 on   (속도↑ · 정확도 영향 거의 없음, Ampere+)
    try:
        torch.set_float32_matmul_precision(
            "high" if deterministic else "medium")
    except AttributeError:
        # < PyTorch-1.12, 옵션 없음
        pass


# ------------------------------------------------------------------ #
# 2. AMP 환경 helper
# ------------------------------------------------------------------ #
def setup_amp_env(enable_amp: bool = True,
                  enable_tf32: bool | None = None
                  ) -> SimpleNamespace:
    """
    • enable_amp : autocast / GradScaler 활성화 여부
    • enable_tf32: True → TF32 on / False → off / None → PyTorch default
    반환:  ns.autocast_ctx, ns.scaler
    """
    if enable_tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32       = enable_tf32

    scaler = GradScaler(enabled=enable_amp and torch.cuda.is_available())
    # autocast(enabled=False) 도 context로 써야 하므로 lambda 래핑
    autocast_ctx = lambda: autocast(enabled=enable_amp)

    return SimpleNamespace(autocast_ctx=autocast_ctx, scaler=scaler)


# ------------------------------------------------------------------ #
# 3. running average
# ------------------------------------------------------------------ #
class AverageMeter:
    """loss / metric 의 running average 계산 용도"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.sum   = 0.0
        self.count = 0
        self.avg   = 0.0

    def update(self, val: float, n: int = 1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / max(self.count, 1e-8)

    def __str__(self):
        return f"{self.avg:.4f}"


# ------------------------------------------------------------------ #
# 4. grad / weight norm
# ------------------------------------------------------------------ #
@torch.no_grad()
def calc_grad_norm(parameters: Iterable[nn.Parameter], norm_type: float = 2.) -> float:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    device = grads[0].device
    total = torch.norm(torch.stack([torch.norm(g, norm_type) for g in grads]), norm_type)
    return total.item()


@torch.no_grad()
def calc_weight_norm(parameters: Iterable[nn.Parameter], norm_type: float = 2.) -> float:
    params = list(parameters)
    if not params:
        return 0.0
    device = params[0].device
    total = torch.norm(torch.stack([torch.norm(p.data, norm_type) for p in params]), norm_type)
    return total.item()


# ------------------------------------------------------------------ #
# 5. checkpointing
# ------------------------------------------------------------------ #
def _make_ckpt_dict(model: nn.Module,
                    optimizer: optim.Optimizer | None = None,
                    scheduler: optim.lr_scheduler._LRScheduler | None = None,
                    scaler: GradScaler | None = None,
                    extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    ckpt = dict(model=model.state_dict())
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    if extra:
        ckpt.update(extra)
    return ckpt


def save_checkpoint(out_path: Path | str,
                    model: nn.Module,
                    optimizer: optim.Optimizer | None = None,
                    scheduler: optim.lr_scheduler._LRScheduler | None = None,
                    scaler: GradScaler | None = None,
                    epoch: int | None = None,
                    **extra):
    """
    model / optimizer / scheduler / scaler state_dict 를 한 파일에 저장.
    FP16 학습 시 GradScaler 도 같이 저장하면 학습 재시작 시 유리.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = _make_ckpt_dict(model, optimizer, scheduler, scaler,
                           {"epoch": epoch, **extra})
    torch.save(ckpt, out_path)
    logging.info(f"✓ checkpoint saved → {out_path}")


def load_checkpoint(ckpt_path: Path | str,
                    model: nn.Module,
                    optimizer: optim.Optimizer | None = None,
                    scheduler: optim.lr_scheduler._LRScheduler | None = None,
                    scaler: GradScaler | None = None,
                    strict: bool = True) -> Dict[str, Any]:
    """
    checkpoint 로드 후 model / optimizer / scheduler / scaler 상태 복원.
    반환값: {'epoch': 마지막 epoch, ...}
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=strict)
    if optimizer   is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler   is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler      is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    logging.info(f"✓ checkpoint loaded ← {ckpt_path}")
    return {k: v for k, v in ckpt.items()
            if k not in {"model", "optimizer", "scheduler", "scaler"}}
