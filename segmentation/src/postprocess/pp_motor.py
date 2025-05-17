# =====================================================================
#  src/postprocess/pp_motor.py  –  ultra-light patch post-proc
# =====================================================================
from __future__ import annotations
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

# ────────────────────────────────────────────────────────────────────
def _prob_map(logit: torch.Tensor) -> np.ndarray:
    """(C,Z,Y,X) → (Z,Y,X) numpy foreground-prob."""
    if logit.shape[0] == 1:                         # binary
        return torch.sigmoid(logit[0]).cpu().numpy()
    return torch.softmax(logit, 0)[1].cpu().numpy() # ch-1 = FG

# ────────────────────────────────────────────────────────────────────
def post_process_pipeline(
    *,                     # keyword-only
    logits: torch.Tensor | None = None,   # (N,C,Z,Y,X)
    batch_meta: Dict[str,Any] | None = None,
    net_out: Dict[str,Any] | None = None, # ← 옛날 dict 입력
    th: float = 0.3,
    top_k: int = 1,                       # ⭐ 가장 conf 높은 K개만
) -> pd.DataFrame:

    # ---- backward compatibility -----------------------------------
    if net_out is not None:
        logits     = net_out["logits"]
        batch_meta = {"roi_start": net_out["roi_start"],
                      "tomo_id"  : net_out["tomo_id"]}

    if logits is None or batch_meta is None:
        raise ValueError("logits & batch_meta must be supplied.")

    roi_start = (batch_meta["roi_start"].cpu().numpy()
                 if isinstance(batch_meta["roi_start"], torch.Tensor)
                 else np.asarray(batch_meta["roi_start"]))
    tomo_ids  = list(batch_meta["tomo_id"])
    N         = logits.size(0)

    rows: List[Dict[str,Any]] = []

    for i in range(N):
        prob = _prob_map(logits[i])          # (Z,Y,X)
        mask = prob > th
        if not mask.any():
            continue

        # --- 확률 상위 top_k 포인트만 픽업 -------------------------
        flat_idx = np.argpartition(-prob[mask], range(min(top_k, mask.sum())))\
                     [:top_k]
        sel = np.column_stack(np.where(mask))[flat_idx]   # (k,3)  z,y,x

        for zz,yy,xx in sel:
            z_g = int(zz + roi_start[i,0])
            y_g = int(yy + roi_start[i,1])
            x_g = int(xx + roi_start[i,2])
            rows.append(dict(tomo_id=tomo_ids[i],
                             x=x_g, y=y_g, z=z_g,
                             conf=float(prob[zz,yy,xx])))

    if not rows:
        return pd.DataFrame(columns=["tomo_id","x","y","z","conf"])
    return pd.DataFrame(rows)
