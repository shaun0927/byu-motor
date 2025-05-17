# ─── Warm-up + Cosine scheduler ─────────────────────────────────────
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        warmup_learning_rate: float,
        decay_factor: float = 0.01,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.decay_factor = decay_factor
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch                  # 0-based
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                # linear warm-up
                progress = step / max(1, self.warmup_steps)
                lr = self.warmup_learning_rate + progress * (base_lr - self.warmup_learning_rate)
            else:
                # cosine decay
                progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                max_lr, min_lr = base_lr, base_lr * self.decay_factor
                lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
            lrs.append(lr)
        return lrs
