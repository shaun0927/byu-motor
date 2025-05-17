"""
EfficientNet-B3 encoder + FlexibleUNet decoder
──────────────────────────────────────────────
• backbone : efficientnet-b3
• class-imbalance 완화 → 작은 class_weights
"""

import copy, pathlib
from configs.common_config import base_cfg

cfg = copy.deepcopy(base_cfg)

cfg.name       = "motor_effb3"
cfg.output_dir = str(pathlib.Path(cfg.output_dir).parent / cfg.name)

cfg.model       = "mdl_motor_effb3"   # <- src/models/mdl_motor_effb3.py
cfg.backbone    = "efficientnet-b3"
cfg.lvl_weights = [0, 0, 0, 1]

# EfficientNet 은 파라미터 수가 많으니 learning-rate 살짝 낮춤
cfg.batch_size = 2
cfg.lr         = 1e-4
cfg.epochs     = 40
