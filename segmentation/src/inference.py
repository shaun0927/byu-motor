#!/usr/bin/env python
# ====================================================================
#  src/inference.py   –   test *.zarr ➜ submission.csv   (stride-free)
# ====================================================================
from __future__ import annotations
import argparse, logging
from pathlib import Path
from types   import SimpleNamespace
from typing  import List, Dict, Any, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.auto        import tqdm

# ───────── local imports ──────────────────────────────────────────
from datasets.motor_dataset import MotorDataset, val_collate_fn
from models.mdl_motor           import Net as MotorSegNetR34
from models.mdl_motor_effb3     import Net as MotorSegNetB3
from utils.training_utils       import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

# ------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser("BYU-motor inference (stride-free)")
    ap.add_argument("--data_root",  required=True)
    ap.add_argument("--ckpt",       required=True)
    ap.add_argument("--arch",       choices=["resnet34", "effb3"],
                    default="resnet34")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--out_csv",    default="submission.csv")
    ap.add_argument("--th",         type=float, default=0.5,
                    help="probability threshold")
    return ap.parse_args()

# ------------------------------------------------------------------
def build_loader(root: Path, bs: int, cfg_ds: SimpleNamespace) -> DataLoader:
    zarr_files = sorted(root.glob("*.zarr"))
    if not zarr_files:
        raise FileNotFoundError(f"no *.zarr under {root}")

    test_df = pd.DataFrame({
        "tomo_id": [p.stem for p in zarr_files],
        "Motor axis 0": -1, "Motor axis 1": -1, "Motor axis 2": -1,
    })

    ds = MotorDataset(test_df, cfg_ds,
                      data_root=root,
                      mode="test",        # stride 파라미터 제거
                      stride=None)
    ds.df = test_df

    return DataLoader(
        ds,
        batch_size          = bs,
        sampler             = SequentialSampler(ds),
        num_workers         = cfg_ds.num_workers,
        pin_memory          = True,
        prefetch_factor     = cfg_ds.prefetch_factor,
        persistent_workers  = True,
        collate_fn          = val_collate_fn
    )

# ------------------------------------------------------------------
def load_model(arch: str, ckpt_path: Path, device):
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    cfg_m = SimpleNamespace(**ckpt.get("cfg", dict(
        in_channels=1, n_classes=1, backbone=arch,
        class_weights=[1., 1.], lvl_weights=[1,1,1,1])))
    net_cls = MotorSegNetB3 if arch == "effb3" else MotorSegNetR34
    net = net_cls(cfg_m)
    net.load_state_dict(ckpt.get("model", ckpt["state_dict"]), strict=False)
    return net.to(device).to(memory_format=torch.channels_last_3d).eval()

# ------------------------------------------------------------------
@torch.no_grad()
def infer(model, loader, device, th: float) -> pd.DataFrame:
    best_hit: Dict[str, Tuple[float,int,int,int]] = {}

    for batch in tqdm(loader, desc="infer"):
        logits = model({
            "input": batch["image"].to(
                device, non_blocking=True, memory_format=torch.channels_last_3d)
        })["logits"]                                       # (B,1,Z,Y,X)

        prob = torch.sigmoid(logits).squeeze(1)            # (B,Z,Y,X)

        for i, tid in enumerate(batch["tomo_id"]):
            p = prob[i]
            if p.max() < th:
                continue
            zz, yy, xx = (p == p.max()).nonzero(as_tuple=True)
            z, y, x = int(zz[0]), int(yy[0]), int(xx[0])
            conf = float(p.max())
            z += int(batch["roi_start"][i, 0])
            y += int(batch["roi_start"][i, 1])
            x += int(batch["roi_start"][i, 2])

            if (tid not in best_hit) or (conf > best_hit[tid][0]):
                best_hit[tid] = (conf, x, y, z)

    sub_rows: List[Dict[str, Any]] = []
    for tid in loader.dataset.df["tomo_id"].unique():
        if tid in best_hit:
            _, x, y, z = best_hit[tid]
            sub_rows.append(dict(tomo_id=tid,
                                 **{f"Motor axis 0": z,
                                    f"Motor axis 1": y,
                                    f"Motor axis 2": x}))
        else:
            sub_rows.append(dict(tomo_id=tid,
                                 **{f"Motor axis {i}": -1 for i in range(3)}))

    return pd.DataFrame(sub_rows).sort_values("tomo_id").reset_index(drop=True)

# ------------------------------------------------------------------
def main():
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_ds = SimpleNamespace(
        roi_size=(96, 96, 96),
        train_sub_epochs=1,
        seed=args.seed,
        num_workers=8,
        prefetch_factor=4,
    )

    loader = build_loader(Path(args.data_root), args.batch_size, cfg_ds)
    model  = load_model(args.arch, Path(args.ckpt), device)

    sub_df = infer(model, loader, device, th=args.th)
    sub_df.to_csv(args.out_csv, index=False)
    logging.info(f"✓ submission saved → {args.out_csv}")

if __name__ == "__main__":
    main()
