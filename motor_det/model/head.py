import torch
from torch import nn


class PointHead(nn.Module):
    """
    Anchor-free centre-+offset head (stride 2).
    - cls   : 1 channel logits
    - offset: 3 channel Tanh * 2
    """
    def __init__(self, in_channels: int = 64, stride: int = 2):
        super().__init__()
        self.stride = stride

        self.cls_stem = nn.Sequential(
            nn.Conv3d(in_channels, 48, 3, padding=1), nn.SiLU(inplace=True), nn.InstanceNorm3d(48),
            nn.Conv3d(48, 48, 3, padding=1),          nn.SiLU(inplace=True), nn.InstanceNorm3d(48),
        )
        self.cls_head = nn.Conv3d(48, 1, 1)
        nn.init.zeros_(self.cls_head.weight)
        nn.init.constant_(self.cls_head.bias, -4.0)   # Sigmoid ≈ 0.017 초기화

        self.off_stem = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1), nn.SiLU(inplace=True), nn.InstanceNorm3d(16),
            nn.Conv3d(16, 16, 3, padding=1),          nn.SiLU(inplace=True), nn.InstanceNorm3d(16),
        )
        self.off_head = nn.Conv3d(16, 3, 1)
        nn.init.zeros_(self.off_head.weight)
        nn.init.constant_(self.off_head.bias, 0.0)

    def forward(self, x: torch.Tensor):
        cls  = self.cls_head(self.cls_stem(x))
        offs = torch.tanh(self.off_head(self.off_stem(x))) * self.stride
        return cls, offs
