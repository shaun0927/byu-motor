from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from functools import cached_property
from torch.utils.data import DataLoader
from tqdm import tqdm

from motor_det.config import InferenceConfig
from motor_det.data.detection import SlidingWindowDataset
from motor_det.model.net import MotorDetNet
from motor_det.utils.voxel import read_test_ids, voxel_spacing_map


class HannWindow:
    """Cached cosine Hann window and its downsampled version."""

    def __init__(self, window: Tuple[int, int, int], stride: int) -> None:
        self.window = window
        self.stride = stride

    @cached_property
    def full(self) -> np.ndarray:
        return cosine_hann_3d(self.window)

    @cached_property
    def downsampled(self) -> np.ndarray:
        return self.full[:: self.stride, :: self.stride, :: self.stride]


def cosine_hann_3d(shape: Sequence[int]) -> np.ndarray:
    """Return 3-D separable cosine (Hann) window, float32, shape (D,H,W)."""

    def hann(L: int) -> np.ndarray:
        return 0.5 - 0.5 * np.cos(2.0 * math.pi * np.arange(L) / L)

    wz, wy, wx = (hann(L) for L in shape)
    return (wz[:, None, None] * wy[None, :, None] * wx[None, None, :]).astype(np.float32)


@torch.no_grad()
def infer_single_tomo(
    *,
    zarr_path: Path,
    net: torch.nn.Module,
    window: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    stride_head: int,
    spacing_Å: float,
    batch_size: int,
    num_workers: int,
    prob_thr: float,
    sigma_Å: float,
    iou_thr: float,
    device: torch.device,
    early_exit_thr: float | None = None,
    flip_tta: bool = False,
    num_tiles: Tuple[int, int, int] | None = None,
    tile_xy: int | None = None,
) -> np.ndarray:
    """Return predicted motor centre for one tomogram in Å units."""

    ds = SlidingWindowDataset(
        zarr_path,
        window=window,
        stride=stride,
        dtype=np.float32,
        cache_size=64,
        num_tiles=num_tiles,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    D_o, H_o, W_o = (s // stride_head for s in ds.store.shape)
    acc_prob = np.zeros((D_o, H_o, W_o), np.float32)
    acc_offs = np.zeros((3, D_o, H_o, W_o), np.float32)
    acc_w = np.zeros((D_o, H_o, W_o), np.float32)

    hann = HannWindow(window, stride_head).downsampled

    net.eval()
    amp_ctx = torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast

    orientations = [()]
    if flip_tta:
        orientations += [(0,), (1,), (2,)]

    with amp_ctx():
        for batch in tqdm(loader, desc=zarr_path.stem, leave=False):
            patches = batch["image"].to(device, non_blocking=True)
            origins = batch["origin"]
            for axes in orientations:
                p = patches
                if axes:
                    dims = [ax + 2 for ax in axes]
                    p = torch.flip(p, dims=dims)
                out = net(p)
                logits = out["cls"]
                offsets = out["offset"]
                if axes:
                    logits = torch.flip(logits, dims=dims)
                    offsets = torch.flip(offsets, dims=dims)
                    for ax in axes:
                        offsets[:, ax].neg_()

                prob_np = torch.sigmoid(logits).cpu().numpy()[:, 0]
                offs_np = offsets.cpu().numpy()

                B = prob_np.shape[0]
                for b in range(B):
                    oz, oy, ox = (origins[k][b].item() // stride_head for k in range(3))
                    dz, dy, dx = prob_np[b].shape

                    prob_b = prob_np[b, :dz, :dy, :dx]
                    offs_b = offs_np[b, :, :dz, :dy, :dx]
                    hann_bds = hann[:dz, :dy, :dx]

                    if tile_xy is None:
                        slz = slice(oz, oz + dz)
                        sly = slice(oy, oy + dy)
                        slx = slice(ox, ox + dx)

                        acc_prob[slz, sly, slx] += prob_b * hann_bds
                        acc_offs[:, slz, sly, slx] += offs_b * hann_bds
                        acc_w[slz, sly, slx] += hann_bds
                    else:
                        t = int(tile_xy)
                        for zz in range(dz):
                            for yy in range(0, dy, t):
                                for xx in range(0, dx, t):
                                    pr = prob_b[zz, yy : yy + t, xx : xx + t]
                                    of = offs_b[:, zz, yy : yy + t, xx : xx + t]
                                    hw = hann_bds[zz, yy : yy + t, xx : xx + t]

                                    slz = slice(oz + zz, oz + zz + 1)
                                    sly = slice(oy + yy, oy + yy + pr.shape[0])
                                    slx = slice(ox + xx, ox + xx + pr.shape[1])

                                    acc_prob[slz, sly, slx] += pr * hw
                                    acc_offs[:, slz, sly, slx] += of * hw
                                    acc_w[slz, sly, slx] += hw

            if early_exit_thr is not None and acc_prob.max() >= early_exit_thr:
                break

    mask = acc_w > 0
    acc_prob[mask] /= acc_w[mask]
    acc_offs[:, mask] /= acc_w[mask]

    flat_idx = np.argmax(acc_prob)
    z, y, x = np.unravel_index(flat_idx, acc_prob.shape)
    off_x, off_y, off_z = acc_offs[:, z, y, x]

    ctr_voxel = np.array([x, y, z], dtype=np.float32) * stride_head + np.array([off_x, off_y, off_z], dtype=np.float32)
    ctr_A_xyz = ctr_voxel[::-1] * spacing_Å
    return ctr_A_xyz[None]


def run_inference(
    weights: Path | str,
    data_root: Path | str,
    out_csv: Path | str,
    *,
    cfg: InferenceConfig,
    device: torch.device | None = None,
) -> Path:
    """Load ``weights`` and run inference over ``data_root`` test tomograms."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    torch.set_float32_matmul_precision("high")

    net = MotorDetNet()
    ckpt = torch.load(weights, map_location="cpu")
    net.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    net.to(device)

    test_ids = read_test_ids(data_root)
    spacing_map = voxel_spacing_map(data_root)

    rows = []
    for tid in tqdm(test_ids, desc="Tomograms"):
        zp = Path(data_root) / "processed" / "zarr" / "test" / f"{tid}.zarr"
        spacing = spacing_map.get(tid, cfg.default_spacing)
        ctrs = infer_single_tomo(
            zarr_path=zp,
            net=net,
            window=(cfg.win_d, cfg.win_h, cfg.win_w),
            stride=(cfg.stride_d, cfg.stride_h, cfg.stride_w),
            stride_head=cfg.stride_head,
            spacing_Å=spacing,
            batch_size=cfg.batch,
            num_workers=cfg.num_workers,
            prob_thr=cfg.prob_thr,
            sigma_Å=cfg.sigma,
            iou_thr=cfg.iou_thr,
            device=device,
            early_exit_thr=cfg.early_exit,
            flip_tta=cfg.flip_tta,
            num_tiles=(cfg.num_tiles_d, cfg.num_tiles_h, cfg.num_tiles_w)
            if cfg.num_tiles_d is not None
            else None,
            tile_xy=cfg.tile_xy,
        )
        if ctrs.size == 0:
            rows.append([tid, -1, -1, -1])
        else:
            rows.append([tid, *ctrs[0]])

    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2"])
        writer.writerows(rows)
    print("Saved →", out)
    return out


def infer(cfg: InferenceConfig) -> Path:
    return run_inference(cfg.weights, cfg.data_root, cfg.out_csv, cfg=cfg)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Path to config file")
    p.add_argument(
        "--env_prefix",
        type=str,
        default="BYU_INFER_",
        help="Prefix for environment variable overrides",
    )
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    cfg = InferenceConfig.load(args.config, env_prefix=args.env_prefix)
    infer(cfg)


if __name__ == "__main__":
    main()
