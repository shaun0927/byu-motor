"""Top-level package for BYU Motor Detection.

This module exposes the training data module and Lightning module used in
the BYU motor detection pipeline.
"""

from importlib.metadata import version as _version

from .data.module import MotorDataModule
from .engine.lit_module import LitMotorDet

__all__ = ["MotorDataModule", "LitMotorDet"]

__version__ = _version("motor_det")
