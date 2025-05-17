from __future__ import annotations
from typing import Dict, Tuple

import lightning as L
import torch
from torch import nn, Tensor

from motor_det.model.net import MotorDetNet
from motor_det.loss.losses import motor_detection_loss
from motor_det.optim.cosine_with_warmup import WarmupCosineScheduler
from motor_det.postprocess.decoder import decode_with_nms
from motor_det.metrics.det_metric  import fbeta_score


class LitMotorDet(L.LightningModule):
    """
    LightningModule wrapping MotorDetNet + custom BCE + L1 loss.
    """

    def __init__(
        self,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        warmup_steps: int = 500,
        total_steps: int = 30_000,
    ):
        super().__init__()
        self.save_hyperparameters()

        net = MotorDetNet()
        if hasattr(torch, "compile"):
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            try:
                net = torch.compile(net)
            except Exception:
                pass
        self.net: nn.Module = net

    # ------------------------------------------------ #
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return self.net(x)

    # ------------------------------------------------ #
    def _shared_step(self, batch: Dict[str, Tensor], stage: str):
        preds = self(batch["image"])
        loss, logs = motor_detection_loss(preds, batch)
        logs = {f"{stage}/{k}": v for k, v in logs.items()}
        self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "val")

        preds   = self(batch["image"])
        logits  = preds["cls"]
        offsets = preds["offset"]

        centers_pred = decode_with_nms(
            logits, offsets, stride=2,
            prob_thr=0.5, sigma=60.0, iou_thr=0.25
        )[0]

        gt_centers = batch["centers_Å"][0]
        f2, prec, rec, tp, fp, fn = fbeta_score(
            centers_pred, gt_centers, beta=2, dist_thr=1000.0
        )

        # 한 번에 batch-epoch 단위로 모두 기록
        self.log_dict({
            "val/f2":   f2,
            "val/prec": prec,
            "val/rec":  rec,
            "val/tp":   tp,
            "val/fp":   fp,
            "val/fn":   fn,
        }, on_step=False, on_epoch=True, prog_bar=True)

        return {"f2": f2, "prec": prec, "rec": rec, "tp": tp, "fp": fp, "fn": fn}

    # ------------------------------------------------ #
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        sched = WarmupCosineScheduler(
            optimizer=opt,
            warmup_steps=self.hparams.warmup_steps,
            total_steps=self.hparams.total_steps,
            warmup_learning_rate=self.hparams.lr * 0.1,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }
