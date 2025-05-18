import torch
from torch import nn

from motor_det.model.backbone import MotorBackbone
from motor_det.models.detection_head import ObjectDetectionHead


class MotorDetNet(nn.Module):
    """SegResNet backbone with multi-scale FCOS-style heads."""

    def __init__(self):
        super().__init__()
        self.backbone = MotorBackbone()

        # Channel dimensions for the feature maps produced by ``MotorBackbone``
        # are 32, 64 and 128 at strides 4, 8 and 16 respectively.  The detection
        # heads were incorrectly initialised with doubled channel sizes which
        # resulted in shape mismatch errors during training.  Initialise each
        # head with the correct number of input channels.
        self.head_s4 = ObjectDetectionHead(32, 1, stride=4)
        self.head_s8 = ObjectDetectionHead(64, 1, stride=8)
        self.head_s16 = ObjectDetectionHead(128, 1, stride=16)

    @property
    def strides(self):
        return [4, 8, 16]

    def forward(self, x: torch.Tensor):
        f4, f8, f16 = self.backbone.multi_feats(x)

        logits = []
        offsets = []
        for head, feat in zip((self.head_s4, self.head_s8, self.head_s16), (f4, f8, f16)):
            lg, off = head(feat)
            logits.append(lg)
            offsets.append(off)

        return {"logits": logits, "offsets": offsets, "strides": self.strides}
