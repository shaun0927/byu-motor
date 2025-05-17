from dataclasses import dataclass
from motor_det.utils.voxel import DEFAULT_TEST_SPACING

@dataclass
class InferConfig:
    """Configuration defaults for inference."""
    win_d: int = 192
    win_h: int = 128
    win_w: int = 128
    stride_d: int = 96
    stride_h: int = 64
    stride_w: int = 64
    stride_head: int = 2
    batch: int = 1
    num_workers: int = 4
    prob_thr: float = 0.5
    sigma: float = 60.0
    iou_thr: float = 0.25
    default_spacing: float = DEFAULT_TEST_SPACING
    early_exit: float | None = None
