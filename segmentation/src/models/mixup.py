# =====================================================================
#  src/models/mixup.py
# ---------------------------------------------------------------------
#  Beta(β,β) Mix-Up  for 3-D (or N-D) tensors.
#  Supports both continuous interpolation and binary-OR (“add”) modes,
#  while remaining backward-compatible with the old `mixadd=` flag.
# =====================================================================
from __future__ import annotations
from typing import Tuple

import torch
from torch.distributions import Beta


class Mixup:
    """
    Parameters
    ----------
    beta        : float  – Beta distribution parameter (>0).  
                            β ≤ 0 → MixUp disabled (identity).
    mode        : {'interpolate', 'add'}
        * interpolate : λ·A + (1-λ)·B   (continuous labels / heat-maps)
        * add         : clamp(A + B, 0, 1)   (binary masks)
    same_lambda : bool – True → one λ shared across the mini-batch.
    mixadd      : bool – legacy alias for ``mode='add'`` (kept for
                          older config files).  If given it overrides
                          *mode*.
    """

    def __init__(
        self,
        beta: float = 1.0,
        *,
        mode: str = "interpolate",
        same_lambda: bool = False,
        mixadd: bool | None = None,
    ) -> None:
        # ---- backward-compat ---------------------------------------
        if mixadd is not None:                       # old kwarg present
            mode = "add" if mixadd else "interpolate"

        if beta <= 0.0:
            # no mixing – but keep interface identical
            self.enabled = False
        else:
            self.enabled = True
            self.beta_dist = Beta(beta, beta)

        if mode not in ("interpolate", "add"):
            raise ValueError("mode must be 'interpolate' or 'add'")
        self.mode = mode
        self.same_lambda = same_lambda

    # ----------------------------------------------------------------
    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns
        -------
        mixed_x , mixed_y
        """
        # quick exit paths -------------------------------------------
        if (not self.enabled) or x.size(0) < 2:
            return x, y

        B = x.size(0)
        device = x.device
        dtype_x = x.dtype
        dtype_y = y.dtype if y is not None else None

        # sample λ   --------------------------------------------------
        lam = self.beta_dist.sample(
            (1 if self.same_lambda else B,)
        ).to(device)  # (B,) or (1,)
        # shuffle indices  -------------------------------------------
        perm = torch.randperm(B, device=device)

        # reshape λ for broadcasting ---------------------------------
        lam_x = lam.view([B] + [1] * (x.ndim - 1))
        if y is not None:
            lam_y = lam.view([B] + [1] * (y.ndim - 1))

        # mix images --------------------------------------------------
        mixed_x = lam_x * x + (1.0 - lam_x) * x[perm]

        # mix targets -------------------------------------------------
        if y is None:
            return mixed_x.to(dtype_x), None

        if self.mode == "interpolate":
            mixed_y = lam_y * y + (1.0 - lam_y) * y[perm]
        else:  # "add"  – binary OR
            mixed_y = torch.clamp(y + y[perm], 0, 1)

        return mixed_x.to(dtype_x), mixed_y.to(dtype_y)
