# motor_det/engine/infer.py

from __future__ import annotations
import argparse, csv, math
from pathlib import Path
from typing import Tuple, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from motor_det.data.sliding_window import SlidingWindowDataset
from motor_det.model.net import MotorDetNet
from motor_det.postprocess.decoder import decode_with_nms
from motor_det.utils.voxel import (
    voxel_spacing_map,
    read_test_ids,
    DEFAULT_TEST_SPACING,
)
from motor_det.config import InferenceConfig


class HannWindow:
    """Cached cosine Hann window and its downsampled version."""

    def __init__(self, window: Tuple[int, int, int], stride: int) -> None:
        self.window = window
        self.stride = stride

    @torch.cached_property
    def full(self) -> np.ndarray:
        return cosine_hann_3d(self.window)

    @torch.cached_property
    def downsampled(self) -> np.ndarray:
        return self.full[::self.stride, ::self.stride, ::self.stride]


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
    spacing_Å: float,       # Å / voxel
    batch_size: int,
    num_workers: int,
    prob_thr: float,
    sigma_Å: float,
    iou_thr: float,
    device: torch.device,
    early_exit_thr: float | None = None,
    flip_tta: bool = False,
) -> np.ndarray:
    """
    하나의 톰ogram을 sliding-window + Hann 가중치 + NMS로 추론하여
    motor center를 Å 단위 (X,Y,Z) 로 반환합니다.
    ``spacing_Å`` 는 1 voxel 당 Å 길이입니다.
    """
    # ─── DataLoader 준비 ─────────────────────────────────
    ds = SlidingWindowDataset(zarr_path, window=window, stride=stride, dtype=np.float32)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # ─── accumulator 초기화 (output 해상도) ────────────────
    D_o, H_o, W_o = (s // stride_head for s in ds.store.shape)
    acc_prob = np.zeros((D_o, H_o, W_o), np.float32)
    acc_offs = np.zeros((3, D_o, H_o, W_o), np.float32)
    acc_w    = np.zeros((D_o, H_o, W_o), np.float32)

    # ─── Hann 윈도우 한 번만 다운샘플 ───────────────────────
    hann = HannWindow(window, stride_head)
    hann_ds = hann.downsampled  # (D',H',W')

    net.eval()
    amp_ctx = torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast

    # ─── sliding-window 루프 ───────────────────────────────
    orientations = [()]
    if flip_tta:
        orientations += [(0,), (1,), (2,)]

    with amp_ctx():
        stop = False
        for patches, origins in tqdm(loader, desc=zarr_path.stem, leave=False):
            patches = patches.to(device, non_blocking=True) / 255.0
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

                prob_np = torch.sigmoid(logits).cpu().numpy()[:, 0, ...]
                offs_np = offsets.cpu().numpy()

                B = prob_np.shape[0]
                for b in range(B):
                    oz, oy, ox = (origins[k][b].item() // stride_head for k in range(3))
                    dz, dy, dx = prob_np[b].shape

                    prob_b   = prob_np[b, :dz, :dy, :dx]
                    offs_b   = offs_np[b, :, :dz, :dy, :dx]
                    hann_bds = hann_ds[:dz, :dy, :dx]

                    slz = slice(oz, oz + dz)
                    sly = slice(oy, oy + dy)
                    slx = slice(ox, ox + dx)

                    acc_prob [slz, sly, slx]   += prob_b * hann_bds
                    acc_offs[:, slz, sly, slx] += offs_b * hann_bds
                    acc_w   [slz, sly, slx]    += hann_bds

            if early_exit_thr is not None and acc_prob.max() >= early_exit_thr:
                stop = True
                break

    # 6) 전체 가중합 → 평균 (종전 그대로)
    mask = acc_w > 0
    acc_prob[mask]  /= acc_w[mask]
    acc_offs[:,mask] /= acc_w[mask]

    # 7) “가장 높은 확률” 위치 하나만 뽑기
    #─────────────────────────────────────────────────────
    # 7.1) heatmap에서 최대값 인덱스
    flat_idx = np.argmax(acc_prob)
    z, y, x = np.unravel_index(flat_idx, acc_prob.shape)  # downsampled coord

    # 7.2) offset 채널 순서 고정 (X, Y, Z)
    off_x, off_y, off_z = acc_offs[:, z, y, x]

    # 7.3) voxel 좌표로 환산 (downsampled → 원해상도)
    ctr_voxel = np.array([x, y, z], dtype=np.float32) * stride_head \
                + np.array([off_x, off_y, off_z], dtype=np.float32)

    # 7.4) Å 단위로 변환 (1 voxel = spacing_Å Å, 순서 유지: [x, y, z])
    ctr_voxel_xyz = ctr_voxel[::-1]                    # (3,)
    ctr_A_xyz = ctr_voxel_xyz * spacing_Å              # (3,)
    return ctr_A_xyz[None, :]                          # (1,3)


def infer(cfg: InferenceConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    net = MotorDetNet()
    ckpt = torch.load(cfg.weights, map_location="cpu")
    net.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    net.to(device)

    test_ids    = read_test_ids(cfg.data_root)
    spacing_map = voxel_spacing_map(cfg.data_root)

    rows = []
    for tid in tqdm(test_ids, desc="Tomograms"):
        zp      = Path(cfg.data_root) / "processed" / "zarr" / "test" / f"{tid}.zarr"
        spacing = spacing_map.get(tid, cfg.default_spacing)
        ctrs    = infer_single_tomo(
            zarr_path   = zp,
            net         = net,
            window      = (cfg.win_d, cfg.win_h, cfg.win_w),
            stride      = (cfg.stride_d, cfg.stride_h, cfg.stride_w),
            stride_head = cfg.stride_head,
            spacing_Å   = spacing,
            batch_size  = cfg.batch,
            num_workers = cfg.num_workers,
            prob_thr    = cfg.prob_thr,
            sigma_Å     = cfg.sigma,
            iou_thr     = cfg.iou_thr,
            device      = device,
            early_exit_thr = cfg.early_exit,
            flip_tta    = cfg.flip_tta,
        )
        if ctrs.size == 0:
            rows.append([tid, -1, -1, -1])
        else:
            rows.append([tid, *ctrs[0]])

    out = Path(cfg.out_csv)
    out.parent.mkdir(exist_ok=True, parents=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2"])
        writer.writerows(rows)
    print("Saved →", cfg.out_csv)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_csv",  required=True)
    parser.add_argument("--win_d", type=int, default=192)
    parser.add_argument("--win_h", type=int, default=128)
    parser.add_argument("--win_w", type=int, default=128)
    parser.add_argument("--stride_d", type=int, default=96)
    parser.add_argument("--stride_h", type=int, default=64)
    parser.add_argument("--stride_w", type=int, default=64)
    parser.add_argument("--stride_head", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prob_thr", type=float, default=0.50)
    parser.add_argument("--sigma", type=float, default=60.0)
    parser.add_argument("--iou_thr", type=float, default=0.25)
    parser.add_argument(
        "--default_spacing",
        type=float,
        default=DEFAULT_TEST_SPACING,
    )
    parser.add_argument("--early_exit", type=float, default=None)
    parser.add_argument("--flip_tta", action="store_true", help="Enable flip test-time augmentation")

def main() -> None:
    args = parser.parse_args()

    cfg = InferenceConfig.load(args.config, env_prefix=args.env_prefix)
    infer(cfg)


if __name__ == "__main__":
    main()
