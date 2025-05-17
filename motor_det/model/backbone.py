import torch
import monai
from torch import nn


class MotorBackbone(nn.Module):
    """
    3-D SegResNet encoder (stride 32) — feature map C=64
    """
    def __init__(self, in_channels: int = 1, init_filters: int = 32):
        super().__init__()
        self.net = monai.networks.nets.SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=init_filters,      # 최종 conv는 사용 안 함
            init_filters=init_filters,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),            # encoder 전용
            dropout_prob=None,
            use_conv_final=False,
            norm=("INSTANCE", {}),          # ★ GroupNorm → InstanceNorm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net.convInit(x)
        downs = []
        for down in self.net.down_layers:
            x = down(x)
            downs.append(x)
        return downs[1]                     # stride 2, channels 64
