from __future__ import annotations
from typing import Dict, Tuple

import lightning as L
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch import nn, Tensor
from lightning.pytorch.utilities.rank_zero import rank_zero_info

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
        nms_algorithm: str = "vectorized",
        nms_switch_thr: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()

        net = MotorDetNet()
        import torch as _torch  # avoid potential local shadowing issues
        import warnings

        if _torch.cuda.is_available() and hasattr(_torch, "compile"):
            try:
                import triton  # noqa: F401
            except Exception:
                warnings.warn("Triton not found; skipping torch.compile", stacklevel=2)
            else:
                import torch._dynamo
                torch._dynamo.config.suppress_errors = True
                try:
                    net = _torch.compile(net)
                except Exception:
                    warnings.warn("torch.compile failed; continuing without compilation", stacklevel=2)

        self.net: nn.Module = net

    # ------------------------------------------------ #
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return self.net(x)

    # ------------------------------------------------ #
    def _shared_step(self, batch: Dict[str, Tensor], stage: str):
        preds = self(batch["image"])
        loss, logs = motor_detection_loss(preds, batch)
        logs = {f"{stage}/{k}": v for k, v in logs.items()}
        batch_size = batch["image"].size(0)
        self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=True,
                      batch_size=batch_size)

        # Print metrics to the console for real-time monitoring
        msg = (
            f"[{stage.upper()}] step {self.global_step:>6}: "
            + ", ".join(f"{k.split('/')[-1]}={float(v):.4f}" for k, v in logs.items())
        )
        rank_zero_info(msg)

        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def on_validation_epoch_start(self) -> None:
        self._val_outputs: list[dict[str, torch.Tensor]] = []

    def validation_step(self, batch, batch_idx):
        # 1) 기본 손실 및 로그
        loss = self._shared_step(batch, "val")

        # 2) 예측 → NMS
        preds    = self(batch["image"])
        logits   = preds["cls"]
        offsets  = preds["offset"]

        centers_pred = decode_with_nms(
            logits,
            offsets,
            stride=4,
            prob_thr=0.5,
            sigma=60.0,
            iou_thr=0.25,
            algorithm=self.hparams.nms_algorithm,
            switch_thr=self.hparams.nms_switch_thr,
        )[0]

        # 3) voxel → Å 변환
        spacing = float(batch["spacing_Å_per_voxel"][0])
        centers_pred = centers_pred * spacing

        # 4) F2 및 세부 지표
        gt_centers = batch["centers_Å"][0]
        f2, prec, rec, tp, fp, fn = fbeta_score(
            centers_pred, gt_centers, beta=2, dist_thr=1000.0
        )

        # 5) int → Tensor 캐스팅 후 저장
        tp_t = torch.tensor(tp, device=self.device)
        fp_t = torch.tensor(fp, device=self.device)
        fn_t = torch.tensor(fn, device=self.device)
        self._val_outputs.append({"tp": tp_t, "fp": fp_t, "fn": fn_t})

        # (선택) 스텝 단위 로그
        # 각 배치에 대한 세부 지표는 훈련 진행 표시를 지나치게
        # 복잡하게 만들 수 있으므로 더 이상 기록하지 않는다.
        # 필요한 경우 이벤트 파일에서 직접 값을 확인한다.

        # Print step metrics to the console
        rank_zero_info(
            f"[VAL] step {self.global_step:>6}: f2={f2:.4f}, tp={tp}, fp={fp}, fn={fn}"
        )

        return {"tp": tp_t, "fp": fp_t, "fn": fn_t}

    # ------------------------------------------------ #
    def on_validation_epoch_end(self):      #반드시 함수 이름 on_validation_epoch_end(self)이어야 함
        # 0) 빈 리스트면 바로 종료
        if not self._val_outputs:
            return

        device = self.device
        tp = torch.stack([o["tp"] for o in self._val_outputs]).sum().to(device)
        fp = torch.stack([o["fp"] for o in self._val_outputs]).sum().to(device)
        fn = torch.stack([o["fn"] for o in self._val_outputs]).sum().to(device)

        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        beta2 = 4.0
        f2 = (1 + beta2) * prec * rec / (beta2 * prec + rec + 1e-9)

        self.log_dict(
            {
                "val/f2":   f2,
                "val/prec": prec,
                "val/rec":  rec,
                "val/tp":   tp,
                "val/fp":   fp,
                "val/fn":   fn,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        # Print aggregated validation metrics
        rank_zero_info(
            f"[VAL] step {self.global_step:>6}: f2={f2:.4f}, tp={int(tp)}, fp={int(fp)}, fn={int(fn)}"
        )

        # → Lightning 2.x 에서는 반환값이 필요 없으므로 그냥 종료
        self._val_outputs.clear()   # 다음 epoch 대비 초기화

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
