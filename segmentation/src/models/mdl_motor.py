# =====================================================================
#  src/models/mdl_motor.py
# =====================================================================
from __future__ import annotations
from types import SimpleNamespace
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.flexible_unet import (
    FLEXUNET_BACKBONE, UNetDecoder, SegmentationHead,
)

from .mixup  import Mixup
from .losses import DenseCrossEntropy, to_ce_target, \
                    human_format, count_parameters
# ------------------------------------------------------------------ #
# 1. Patched decoder – 모든 stage feature 반환
# ------------------------------------------------------------------ #
class _PatchedDecoder(UNetDecoder):
    def forward(self, feats: List[torch.Tensor], skip_connect: int):
        skips = feats[:-1][::-1]
        feats = feats[1:][::-1]
        outs  = [feats[0]]
        x     = feats[0]
        for i, blk in enumerate(self.blocks):
            x = blk(x, skips[i] if i < skip_connect else None)
            outs.append(x)
        return outs                   # len == encoder_depth-1
# ------------------------------------------------------------------ #
# 2. Main network
# ------------------------------------------------------------------ #
class Net(nn.Module):
    """
    deep-supervision head를 ‘2개’만 사용하도록 경량화한 버전
    ----------------------------------------------------------------
    cfg 필수 항목 (★ = 새 옵션)
        in_channels, n_classes
        mixup_p, mixup_beta, mixadd
        class_weights, lvl_weights  (길이는 자동 맞춰짐)
        backbone='resnet34'
        soft_sigma  (선택)
        ds_heads★   : deep-supervision head 개수 (default=2)
    """

    # --- util: 3-D Gaussian kernel ---------------------------------
    @staticmethod
    def _gauss_kernel(ch: int, sigma: float, dev: torch.device):
        r = int(sigma * 4 + .5)
        ax = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
        zz, yy, xx = torch.meshgrid(ax, ax, ax, indexing="ij")
        k = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
        k = (k / k.max()).unsqueeze(0).unsqueeze(0)       # (1,1,D,H,W)
        return k.repeat(ch, 1, 1, 1, 1)                   # depth-wise conv

    # ----------------------------------------------------------------
    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.cfg = cfg

        # ---------------- encoder -----------------------------------
        meta = FLEXUNET_BACKBONE.register_dict[cfg.backbone]
        self.encoder = meta["type"](
            cfg.backbone, spatial_dims=3,
            in_channels=cfg.in_channels, pretrained=False
        )
        enc_feats   = list(meta["feature_channel"])       # [64,128,256,512]
        enc_channels = (cfg.in_channels, *enc_feats)
        enc_depth    = len(enc_channels)

        # ---------------- decoder -----------------------------------
        dec_ch_full    = tuple(enc_feats[::-1])           # 512,256,128,64…
        self.skip_connect = enc_depth - 2
        self.decoder = _PatchedDecoder(
            3, enc_channels, dec_ch_full,
            act=("relu", {"inplace": True}),
            norm=("batch", {"eps": 1e-3, "momentum": .1}),
            dropout=0.0, bias=False,
            upsample="nontrainable", pre_conv="default",
            interp_mode="nearest", align_corners=None, is_pad=True,
        )

        # -------- deep-supervision head 개수 ↓ ----------------------
        ds_heads = int(getattr(cfg, "ds_heads", 2))       # ★ default=2
        seg_ch   = dec_ch_full[-ds_heads:]                # 예: (128, 64)
        self.seg_heads = nn.ModuleList(
            [SegmentationHead(3, ch, cfg.n_classes + 1, 3) for ch in seg_ch]
        )

        # ---------------- misc / loss ------------------------------
        self.mixup   = Mixup(cfg.mixup_beta,
                             mode="add" if getattr(cfg, "mixadd", False)
                                        else "interpolate")
        self.loss_fn = DenseCrossEntropy(torch.as_tensor(cfg.class_weights,
                                                         dtype=torch.float32))
        # lvl_weights 길이를 head 수로 자르기
        full_w = torch.as_tensor(cfg.lvl_weights, dtype=torch.float32)
        self.lvl_w = full_w[: ds_heads]                    # (ds_heads,)

        self.soft_sigma = getattr(cfg, "soft_sigma", None)
        self.register_buffer("_gkern", torch.empty(0), persistent=False)

        print(f"Net parameters: {human_format(count_parameters(self))}")

    # ----------------------------------------------------------------
    def forward(self, batch):
        x = batch["input"]
        y = batch.get("target")

        # ---------- MixUp ------------------------------------------
        if self.training and y is not None and \
           torch.rand(1, device=x.device) < self.cfg.mixup_p:
            x, y = self.mixup(x, y)

        # ---------- encoder / decoder ------------------------------
        feats      = self.encoder(x)
        dec_feats  = self.decoder(feats, self.skip_connect)
        dec_feats  = dec_feats[-len(self.seg_heads):]      # ← tail K stages
        outs       = [head(dec_feats[i])
                      for i, head in enumerate(self.seg_heads)]

        out_dict: dict = {}

        # ---------- loss -------------------------------------------
        if y is not None:
            if self.soft_sigma:
                if self._gkern.numel() == 0 or self._gkern.device != y.device:
                    self._gkern = self._gauss_kernel(1, self.soft_sigma, y.device)
                pad   = self._gkern.shape[-1] // 2
                y_soft = F.conv3d(y, self._gkern, padding=pad, groups=1).clamp(0, 1)
            else:
                y_soft = y

            ys  = [F.adaptive_max_pool3d(y_soft, o.shape[-3:]) for o in outs]
            per = torch.stack([self.loss_fn(outs[i], to_ce_target(ys[i]))[0]
                               for i in range(len(outs))])           # (ds_heads,)
            w   = self.lvl_w.to(per.device)
            out_dict["loss"] = (per * w).sum() / w.sum()

        # ---------- inference output -------------------------------
        if not self.training:
            out_dict |= {
                "logits"    : outs[-1],
                "roi_start" : batch["roi_start"],
            }
        return out_dict
