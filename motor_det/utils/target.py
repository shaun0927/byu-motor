import numpy as np
import torch


def build_target_maps(
    centers_voxel: np.ndarray,
    crop_size: tuple[int, int, int],
    stride: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    centers_voxel : (N,3) ndarray  -- voxel 좌표 (x,y,z) 단위
    crop_size     : (D,H,W)        -- 원본 crop 크기
    stride        : int            -- 출력 해상도 축소비 (default 2)

    Returns
    -------
    cls_map  : (1, D/stride, H/stride, W/stride) float32
    offset   : (3, D/stride, H/stride, W/stride) float32
               (Δx, Δy, Δz) — 실제 중심 - grid*stride,  [-stride/2, +stride/2]
    """
    d, h, w = crop_size
    dz, dy, dx = d // stride, h // stride, w // stride

    cls_map = np.zeros((1, dz, dy, dx), dtype=np.float32)
    off_map = np.zeros((3, dz, dy, dx), dtype=np.float32)

    if centers_voxel.size == 0:
        return cls_map, off_map

    # grid index = floor(center / stride)
    grid = np.floor_divide(centers_voxel, stride).astype(int)     # (N,3)
    offset = centers_voxel - grid * stride                        # (N,3)

    for (gx, gy, gz), (ox, oy, oz) in zip(grid, offset):
        if (0 <= gz < dz) and (0 <= gy < dy) and (0 <= gx < dx):
            cls_map[0, gz, gy, gx] = 1.0
            off_map[:, gz, gy, gx] = (ox, oy, oz)

    return cls_map, off_map


def build_target_maps_torch(
    centers_voxel: torch.Tensor,
    crop_size: tuple[int, int, int],
    stride: int = 2,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch/CUDA version of :func:`build_target_maps`."""
    d, h, w = crop_size
    dz, dy, dx = d // stride, h // stride, w // stride

    cls_map = torch.zeros((1, dz, dy, dx), dtype=torch.float32, device=device)
    off_map = torch.zeros((3, dz, dy, dx), dtype=torch.float32, device=device)

    if centers_voxel.numel() == 0:
        return cls_map, off_map

    grid = torch.div(centers_voxel, stride, rounding_mode="floor").long()
    offset = centers_voxel - grid * stride

    mask = (
        (grid[:, 2] >= 0)
        & (grid[:, 2] < dz)
        & (grid[:, 1] >= 0)
        & (grid[:, 1] < dy)
        & (grid[:, 0] >= 0)
        & (grid[:, 0] < dx)
    )

    grid = grid[mask]
    offset = offset[mask]

    if grid.numel() == 0:
        return cls_map, off_map

    cls_map[0, grid[:, 2], grid[:, 1], grid[:, 0]] = 1.0
    off_map[:, grid[:, 2], grid[:, 1], grid[:, 0]] = offset.t()

    return cls_map, off_map
