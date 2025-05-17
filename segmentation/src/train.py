# =====================================================================
#  src/train.py  ― single-GPU trainer  (stride 로직 완전 제거 버전)
# =====================================================================
from __future__ import annotations
import time, logging
from pathlib import Path
from types   import SimpleNamespace
from typing  import Dict, List, Tuple

import torch
from torch.cuda.amp   import autocast, GradScaler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm.auto        import tqdm
import pandas as pd
import numpy as np

# ─────────── local modules ──────────────────────────────────────────
from datasets.motor_dataset  import MotorDataset, tr_collate_fn, val_collate_fn
from datasets.sampler        import PosBgSampler
from models.mdl_motor        import Net as MotorSegNetR34
from models.mdl_motor_effb3  import Net as MotorSegNetB3
from metrics.metric_motor    import score as fb_score
from utils.training_utils    import set_seed, AverageMeter, save_checkpoint

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")

# ------------------------------------------------------------------ #
# 1. DataLoader factory  (stride 인자 완전 삭제)
# ------------------------------------------------------------------ #
def build_loader(csv_path: str | Path,
                 data_root: str | Path,
                 batch_size: int,
                 *,
                 mode: str,
                 fold_id: int,
                 cfg):

    df_all = pd.read_csv(csv_path)
    df     = df_all[df_all["fold"] != fold_id] if mode == "train" \
           else df_all[df_all["fold"] == fold_id]

    ds = MotorDataset(df, cfg,
                      data_root=data_root,
                      mode=mode,
                      aug=None)
    ds.df = df.copy()

    if mode == "train" and getattr(cfg, "train_sampler", "oversample") == "oversample":
        sampler = PosBgSampler(ds, pos_ratio=getattr(cfg, "pos_ratio", 0.8))
    else:
        sampler = RandomSampler(ds) if mode == "train" else SequentialSampler(ds)

    collate = tr_collate_fn if mode == "train" else val_collate_fn

    return DataLoader(
        ds,
        batch_size          = batch_size,
        sampler             = sampler,
        num_workers         = getattr(cfg, "num_workers", 8),
        pin_memory          = True,
        prefetch_factor     = getattr(cfg, "prefetch_factor", 2),
        persistent_workers  = True,
        collate_fn          = collate,
        drop_last           = (mode == "train"),
    )

# ------------------------------------------------------------------ #
# 2. Model factory
# ------------------------------------------------------------------ #
def build_model(arch: str, cfg):
    return MotorSegNetB3(cfg) if arch.lower() == "effb3" else MotorSegNetR34(cfg)

# ------------------------------------------------------------------ #
# 3-A. training epoch (동일)
# ------------------------------------------------------------------ #
def train_epoch(model, loader, optimizer, scaler,
                device, *, amp: bool, epoch: int):
    model.train()
    meter = AverageMeter()

    pbar = tqdm(loader, desc=f"Train E{epoch}", leave=False)
    for batch in pbar:
        imgs = batch["image"].to(device,
                                 non_blocking=True,
                                 memory_format=torch.channels_last_3d)
        gts  = batch["mask"].to(device, non_blocking=True)

        with autocast(enabled=amp):
            loss = model({
                "input"     : imgs,
                "target"    : gts,
                "roi_start" : batch["roi_start"].to(device),
                "tomo_id"   : batch["tomo_id"],
            })["loss"]

        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        meter.update(loss.item(), imgs.size(0))
        pbar.set_postfix(loss=f"{meter.avg:.4f}")
    return meter.avg

# ------------------------------------------------------------------ #
# 3-B. evaluation  (train 과 동일하게 “최대 conf 1 점”만 선택)
# ------------------------------------------------------------------ #
@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()

    best_hit: Dict[str, Tuple[float,int,int,int]] = {}  # tid → (conf,x,y,z)

    for batch in tqdm(loader, desc="Val", leave=False):
        imgs = batch["image"].to(device, non_blocking=True,
                                 memory_format=torch.channels_last_3d)

        logits = model({"input": imgs})["logits"]          # (B,1,Z,Y,X)
        prob   = torch.sigmoid(logits).squeeze(1)          # (B,Z,Y,X)

        for i, tid in enumerate(batch["tomo_id"]):
            p   = prob[i]
            if p.max() < getattr(cfg, "th_motor", 0.5):
                continue
            zz,yy,xx = (p == p.max()).nonzero(as_tuple=True)
            # 여러 점일 때 첫 번째
            z,y,x = int(zz[0]), int(yy[0]), int(xx[0])
            conf  = float(p.max())
            # global 좌표 변환
            z += int(batch["roi_start"][i,0])
            y += int(batch["roi_start"][i,1])
            x += int(batch["roi_start"][i,2])

            if (tid not in best_hit) or (conf > best_hit[tid][0]):
                best_hit[tid] = (conf,x,y,z)

    # DataFrame 구성
    pred_rows = []
    for tid in loader.dataset.df["tomo_id"].unique():
        if tid in best_hit:
            _,x,y,z = best_hit[tid]
            pred_rows.append(dict(tomo_id=tid,
                                  **{f"Motor axis 0": z,
                                     f"Motor axis 1": y,
                                     f"Motor axis 2": x}))
        else:
            pred_rows.append(dict(tomo_id=tid,
                                  **{f"Motor axis {i}": -1 for i in range(3)}))

    pred_df = pd.DataFrame(pred_rows).sort_values("tomo_id").reset_index(drop=True)
    gt_df   = loader.dataset.df.drop_duplicates("tomo_id") \
                               .sort_values("tomo_id").reset_index(drop=True)
    return fb_score(gt_df, pred_df)

# ------------------------------------------------------------------ #
# 4. Public API (기존과 동일)
# ------------------------------------------------------------------ #
def train(cfg: SimpleNamespace | Dict):
    if isinstance(cfg, dict):
        cfg = SimpleNamespace(**cfg)

    for k in ["data_root","train_csv","arch","batch_size",
              "epochs","lr","seed","fold","output_dir"]:
        if not hasattr(cfg, k):
            raise AttributeError(f"cfg 누락: {k}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tr_loader = build_loader(cfg.train_csv, cfg.data_root, cfg.batch_size,
                             mode="train", fold_id=cfg.fold, cfg=cfg)
    va_loader = build_loader(cfg.train_csv, cfg.data_root, cfg.batch_size,
                             mode="val",   fold_id=cfg.fold, cfg=cfg)

    model     = build_model(cfg.arch, cfg).to(device) \
                .to(memory_format=torch.channels_last_3d)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler    = GradScaler(enabled=getattr(cfg, "amp", False))

    best_f = -1.0
    for ep in range(1, cfg.epochs + 1):
        tic = time.time()
        tr_loss = train_epoch(model, tr_loader, optimizer, scaler,
                              device, amp=getattr(cfg, "amp", False), epoch=ep)
        fbeta   = evaluate(model, va_loader, device, cfg)
        logging.info(f"[E{ep}] loss {tr_loss:.4f} | Fβ {fbeta:.3f} "
                     f"| {time.time()-tic:.1f}s")

        if fbeta > best_f:
            best_f = fbeta
            save_checkpoint(out_dir / "best.pth",
                            model, optimizer, epoch=ep, best_f=best_f)

    logging.info(f"✓ finished – best Fβ {best_f:.3f}")
    return best_f

# ------------------------------------------------------------------ #
# 5. CLI  (val_stride 옵션 삭제)
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("BYU-motor trainer")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--arch", choices=["resnet34","effb3"], default="resnet34")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--output_dir", default="./outputs")
    ap.add_argument("--train_sampler", choices=["oversample","random"],
                    default="oversample")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--pos_ratio", type=float, default=0.8)
    ap.add_argument("--th_motor", type=float, default=0.5)
    cfg_cli = vars(ap.parse_args())

    train(cfg_cli)
