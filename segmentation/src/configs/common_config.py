# =====================================================================
#  src/configs/common_config.py
# ---------------------------------------------------------------------
#  모든 실험 스크립트(cfg_*.py)가 deepcopy 해서 사용하는 **공통 베이스**
#     from configs.common_config import base_cfg
#     cfg = copy.deepcopy(base_cfg)
# =====================================================================
from __future__ import annotations
from types import SimpleNamespace
import pathlib

# ───────────────── 프로젝트 루트 ────────────────────────────────────
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]            # …/byu-motor
_DATA_ROOT    = _PROJECT_ROOT / "data"
_OUTPUT_ROOT  = _PROJECT_ROOT / "outputs"

# ------------------------------------------------------------------ #
#  베이스 딕셔너리
# ------------------------------------------------------------------ #
_base_cfg: dict = {
    # ───── Stage flag / seed ───────────────────────────────────────
    "train"           : True,
    "val"             : True,
    "test"            : False,
    "seed"            : 42,
    "mixed_precision" : True,    # AMP(fp16)
    "bf16"            : False,   # Ampere 이상 GPU → bf16 사용 시 True

    # ───── Data & I/O ──────────────────────────────────────────────
    "data_folder"     : str(_DATA_ROOT),                        # raw / processed 하위
    "train_df"        : str(_DATA_ROOT / "raw" / "train_labels.csv"),
    "val_df"          : None,                                   # fold 로 자동 분리
    "test_df"         : None,
    "data_root"       : str(_DATA_ROOT / "processed" / "train"),# Zarr root
    "fold"            : 0,                                      # 0-4 StratifiedKFold

    # --- DataLoader 설정 -----------------------------------------
    "batch_size"      : 4,
    "batch_size_val"  : None,          # None → train 과 동일
    "num_workers"     : 12,
    "prefetch_factor" : 4,
    "pin_memory"      : True,

    # --- Patch / Sampler -----------------------------------------
    "roi_size"        : (96, 96, 96),   # (z,y,x)
    "train_sub_epochs": 1,             # 1 volume 당 랜덤 patch 개수
    "train_sampler"   : "oversample",   # "oversample" | "random"
    "pos_ratio"       : 0.8,            # oversample 시 positive 비율 목표

    # ───── Model / Loss ───────────────────────────────────────────
    "model"           : "mdl_motor",    # src/models/mdl_motor.py
    "backbone"        : "resnet34",     # "efficientnet-b3" 등으로 교체 가능
    "in_channels"     : 1,
    "classes"         : ["motor"],
    "n_classes"       : 1,

    "class_weights"   : [256.0, 1.0],   # [fg, bg]
    "lvl_weights"     : [1.0, 1.0, 1.0, 1.0],  # deep-sup loss 가중치

    # --- MixUp & Soft-Gaussian label -----------------------------
    "mixup_p"         : 0.3,
    "mixup_beta"      : 1.0,
    "mixadd"          : True,           # True → mask OR 방식
    "soft_sigma"      : 4.0,            # σ(voxel) for gaussian label

    # --- Post-process --------------------------------------------
    "th_motor"        : 0.5,            # foreground threshold

    # ───── Metric (validation) ───────────────────────────────────
    "min_radius"      : 1000.0,         # Å (공식 metric half-diameter)
    "fbeta"           : 2.0,
    "metric_use_kdtree": False,

    # ───── Optimisation ──────────────────────────────────────────
    "optimizer"       : "AdamW",
    "lr"              : 2e-4,
    "weight_decay"    : 1e-4,
    "schedule"        : "cosine",       # cosine / linear / steplr …
    "num_cycles"      : 0.5,
    "epochs"          : 10,             # 스모크 테스트 기본 (실험 시 30–50)
    "grad_accumulation": 1,
    "clip_grad"       : 0.0,

    # ───── Logging / Output ─────────────────────────────────────
    "name"            : "motor_exp",
    "output_dir"      : str(_OUTPUT_ROOT / "motor_exp"),
    "use_tensorboard" : True,
    "disable_tqdm"    : False,
}

# SimpleNamespace 로 변환 후 외부에 제공
base_cfg: SimpleNamespace = SimpleNamespace(**_base_cfg)
