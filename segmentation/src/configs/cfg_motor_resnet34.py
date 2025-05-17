"""
ResNet-34 encoder + FlexibleUNet decoder
────────────────────────────────────────
• backbone : resnet34
• loss     : DenseCrossEntropy  (lvl_weights=[0,0,0,1])
"""

import copy, pathlib
from configs.common_config import base_cfg

cfg = copy.deepcopy(base_cfg)

# 실험 이름 및 출력 폴더
cfg.name       = "motor_resnet34"
cfg.output_dir = str(pathlib.Path(cfg.output_dir).parent / cfg.name)

# 모델 파일 및 backbone 지정
cfg.model         = "mdl_motor"            # <- src/models/mdl_motor.py
cfg.backbone      = "resnet34"
cfg.lvl_weights   = [0, 0, 0, 1]           # penultimate-only loss

# 하이퍼 파라미터(예시)
cfg.batch_size    = 4
cfg.lr            = 2e-4
cfg.epochs        = 40
