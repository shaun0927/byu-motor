import torch
from torch import nn

from motor_det.model.backbone import MotorBackbone
from motor_det.model.head import PointHead


class MotorDetNet(nn.Module):
    """
    backbone(stride 32) → feature(stride 2) → CLS & OFFSET
    출력 dict keys:  'cls', 'offset'
    """
    def __init__(self):
        super().__init__()
        self.backbone = MotorBackbone()
        self.head     = PointHead()

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)          # (B,64,D/2,H/2,W/2)
        cls, off = self.head(feat)
        return {"cls": cls, "offset": off}
