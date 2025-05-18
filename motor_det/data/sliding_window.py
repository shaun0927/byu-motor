from __future__ import annotations
from pathlib import Path
from typing import Generator, Tuple

import torch
from torch.utils.data import DataLoader

from .detection.sliding_window_dataset import SlidingWindowDataset


@torch.no_grad()
def patch_iter(
    zarr_path: Path | str,
    *,
    window: Tuple[int, int, int] = (192, 128, 128),
    stride: Tuple[int, int, int] = (96, 64, 64),
    batch_size: int = 1,
    num_workers: int = 0,
    preload_volume: bool = False,
) -> Generator[tuple[torch.Tensor, tuple[int, int, int]], None, None]:
    """Yield patches and their origins using a DataLoader."""

    ds = SlidingWindowDataset(
        zarr_path,
        window=window,
        stride=stride,
        preload_volume=preload_volume,
        cache_size=0,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    for batch in loader:
        images = batch["image"]
        origins = batch["origin"]
        for i in range(images.size(0)):
            yield images[i], (
                origins[0][i].item(),
                origins[1][i].item(),
                origins[2][i].item(),
            )


@torch.no_grad()
def sliding_window_inference(
    model: torch.nn.Module,
    zarr_path: Path | str,
    *,
    window: Tuple[int, int, int] = (192, 128, 128),
    stride: Tuple[int, int, int] = (96, 64, 64),
    batch_size: int = 1,
    num_workers: int = 0,
    device: torch.device | str | None = None,
    preload_volume: bool = False,
):
    """Run sliding window inference using the given ``model``."""

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else torch.device(device)
    )
    model = model.to(device)
    model.eval()

    for patch, origin in patch_iter(
        zarr_path,
        window=window,
        stride=stride,
        batch_size=batch_size,
        num_workers=num_workers,
        preload_volume=preload_volume,
    ):
        patch = patch.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            out = model(patch.unsqueeze(0))
        yield out, origin

