# ======================================================================
#  src/models/mdl_motor_effb3.py
# ----------------------------------------------------------------------
#  Same architecture as mdl_motor.py but backbone = efficientnet-b3
# ======================================================================
from __future__ import annotations
from types import SimpleNamespace

import torch.nn as nn
from monai.networks.nets.flexible_unet import FLEXUNET_BACKBONE

from .mdl_motor import Net as _BaseNet


class Net(_BaseNet):
    """Simply override backbone name at init."""

    def __init__(self, cfg: SimpleNamespace):
        cfg = SimpleNamespace(**cfg.__dict__)    # shallow copy
        cfg.backbone = "efficientnet-b3"
        super().__init__(cfg)
        # sanity
        assert cfg.backbone in FLEXUNET_BACKBONE.register_dict
