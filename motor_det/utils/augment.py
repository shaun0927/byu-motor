# motor_det/utils/augment.py
import numpy as np
import torch


def random_flip3d(volume: np.ndarray, cls_map: np.ndarray, off_map: np.ndarray):
    """
    Random 3-axis flip for NumPy volumes.
      volume : (D, H, W)   uint8
      cls_map: (1, D/2, H/2, W/2)
      off_map: (3, D/2, H/2, W/2)
    Returns flipped copies (np.ndarray).
    """
    # spatial axis 0-z,1-y,2-x   →  cls/off 는 채널이 1개 더 있으므로 +1
    for ax in (0, 1, 2):
        if np.random.rand() < 0.5:
            volume = np.flip(volume, axis=ax)
            cls_map = np.flip(cls_map, axis=ax + 1)
            off_map = np.flip(off_map, axis=ax + 1)
            # 해당 축 오프셋 부호 반전
            off_map[ax] *= -1

    # np.flip() 가 view 를 반환할 수 있어 .copy() 로 메모리 연속화
    return volume.copy(), cls_map.copy(), off_map.copy()


def random_crop_around_point(
    volume: np.ndarray,
    center: np.ndarray,
    crop_size: tuple[int, int, int],
    *,
    jitter: int = 8,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Crop a patch around ``center`` with jitter.

    Parameters
    ----------
    volume : ``(Z, Y, X)`` ndarray
    center : ``(x, y, z)`` coordinate
    crop_size : desired output size ``(D, H, W)``
    jitter : max random shift in voxels
    """

    D, H, W = crop_size
    Z, Y, X = volume.shape

    # Jitter crop centre then compute the starting coordinates.  ``ctr`` is kept
    # as float so that rounding happens only once when casting to ``int``.

    ctr = center.astype(float)
    if jitter:
        ctr += np.random.randint(-jitter, jitter + 1, size=3)

    start = ctr[[2, 1, 0]] - np.array(crop_size) / 2  # (z,y,x) order
    start = np.round(start).astype(int)
    start = np.clip(start, 0, np.maximum(0, np.array([Z - D, Y - H, X - W])))

    end = start + np.array(crop_size)

    # Ensure that the original ``center`` is inside the crop after clamping.
    cz, cy, cx = center[2], center[1], center[0]
    if not (start[0] <= cz < end[0]):
        start[0] = int(np.clip(cz - D // 2, 0, max(0, Z - D)))
    if not (start[1] <= cy < end[1]):
        start[1] = int(np.clip(cy - H // 2, 0, max(0, Y - H)))
    if not (start[2] <= cx < end[2]):
        start[2] = int(np.clip(cx - W // 2, 0, max(0, X - W)))

    end = start + np.array(crop_size)

    z0, y0, x0 = start
    z1, y1, x1 = end

    patch = np.asarray(volume[z0:z1, y0:y1, x0:x1], dtype=np.uint8)
    return patch, (z0, y0, x0)


def random_erase3d(volume: np.ndarray, max_ratio: float = 0.3) -> np.ndarray:
    """Randomly erase a cuboid region in ``volume``."""
    if np.random.rand() < 0.5:
        D, H, W = volume.shape
        ratio = np.random.uniform(0.1, max_ratio)
        dz, dy, dx = (
            max(1, int(D * ratio)),
            max(1, int(H * ratio)),
            max(1, int(W * ratio)),
        )
        z0 = np.random.randint(0, max(1, D - dz + 1))
        y0 = np.random.randint(0, max(1, H - dy + 1))
        x0 = np.random.randint(0, max(1, W - dx + 1))
        volume[z0 : z0 + dz, y0 : y0 + dy, x0 : x0 + dx] = 0
    return volume


def random_gaussian_noise(volume: np.ndarray, std: float = 5.0) -> np.ndarray:
    """Additive Gaussian noise with probability 0.5."""
    if np.random.rand() < 0.5:
        noise = np.random.normal(0.0, std, size=volume.shape)
        volume = volume.astype(np.float32) + noise
        volume = np.clip(volume, 0, 255).astype(np.uint8)
    return volume


def random_flip3d_torch(
    volume: torch.Tensor, cls_map: torch.Tensor, off_map: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch version of :func:`random_flip3d` operating on CUDA tensors."""
    for ax in (0, 1, 2):
        if torch.rand(1, device=volume.device) < 0.5:
            volume = torch.flip(volume, dims=(ax,))
            cls_map = torch.flip(cls_map, dims=(ax + 1,))
            off_map = torch.flip(off_map, dims=(ax + 1,))
            off_map[ax].neg_()
    return volume.contiguous(), cls_map.contiguous(), off_map.contiguous()


def random_erase3d_torch(volume: torch.Tensor, max_ratio: float = 0.3) -> torch.Tensor:
    if torch.rand(1, device=volume.device) < 0.5:
        D, H, W = volume.shape
        ratio = float(torch.empty(1).uniform_(0.1, max_ratio))
        dz = max(1, int(D * ratio))
        dy = max(1, int(H * ratio))
        dx = max(1, int(W * ratio))
        z0 = int(torch.randint(0, max(1, D - dz + 1), (1,)))
        y0 = int(torch.randint(0, max(1, H - dy + 1), (1,)))
        x0 = int(torch.randint(0, max(1, W - dx + 1), (1,)))
        volume[z0 : z0 + dz, y0 : y0 + dy, x0 : x0 + dx] = 0
    return volume


def random_gaussian_noise_torch(volume: torch.Tensor, std: float = 5.0) -> torch.Tensor:
    if torch.rand(1, device=volume.device) < 0.5:
        noise = torch.normal(0.0, std, size=volume.shape, device=volume.device)
        volume = volume.float() + noise
        volume.clamp_(0, 255)
    return volume



def mixup3d(vol_a: np.ndarray, cls_a: np.ndarray, off_a: np.ndarray,
            vol_b: np.ndarray, cls_b: np.ndarray, off_b: np.ndarray,
            alpha: float = 0.2):
    """Apply MixUp to two 3-D samples."""
    lam = np.random.beta(alpha, alpha)
    vol = vol_a.astype(np.float32) * lam + vol_b.astype(np.float32) * (1.0 - lam)
    vol = np.clip(vol, 0, 255).astype(np.uint8)
    cls = cls_a * lam + cls_b * (1.0 - lam)
    off = off_a * lam + off_b * (1.0 - lam)
    return vol, cls, off


def cutmix3d(vol_a: np.ndarray, cls_a: np.ndarray, off_a: np.ndarray,
             vol_b: np.ndarray, cls_b: np.ndarray, off_b: np.ndarray,
             alpha: float = 1.0):
    """Apply CutMix to two 3-D samples."""
    lam = np.random.beta(alpha, alpha)
    D, H, W = vol_a.shape
    cut_ratio = (1.0 - lam) ** (1.0 / 3.0)
    dz = max(1, int(D * cut_ratio))
    dy = max(1, int(H * cut_ratio))
    dx = max(1, int(W * cut_ratio))
    z0 = np.random.randint(0, max(1, D - dz + 1))
    y0 = np.random.randint(0, max(1, H - dy + 1))
    x0 = np.random.randint(0, max(1, W - dx + 1))
    z1, y1, x1 = z0 + dz, y0 + dy, x0 + dx
    vol = vol_a.copy()
    vol[z0:z1, y0:y1, x0:x1] = vol_b[z0:z1, y0:y1, x0:x1]
    cls = cls_a.copy()
    off = off_a.copy()
    gz0, gy0, gx0 = z0 // 2, y0 // 2, x0 // 2
    gz1, gy1, gx1 = z1 // 2, y1 // 2, x1 // 2
    cls[:, gz0:gz1, gy0:gy1, gx0:gx1] = cls_b[:, gz0:gz1, gy0:gy1, gx0:gx1]
    off[:, gz0:gz1, gy0:gy1, gx0:gx1] = off_b[:, gz0:gz1, gy0:gy1, gx0:gx1]
    return vol, cls, off


def mixup3d_torch(vol_a: torch.Tensor, cls_a: torch.Tensor, off_a: torch.Tensor,
                  vol_b: torch.Tensor, cls_b: torch.Tensor, off_b: torch.Tensor,
                  alpha: float = 0.2):
    """CUDA version of :func:`mixup3d`."""
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    vol = vol_a.float().mul(lam).add_(vol_b.float(), alpha=1.0 - lam)
    vol.clamp_(0, 255)
    vol = vol.type_as(vol_a)
    cls = cls_a.mul(lam).add_(cls_b, alpha=1.0 - lam)
    off = off_a.mul(lam).add_(off_b, alpha=1.0 - lam)
    return vol, cls, off


def cutmix3d_torch(vol_a: torch.Tensor, cls_a: torch.Tensor, off_a: torch.Tensor,
                   vol_b: torch.Tensor, cls_b: torch.Tensor, off_b: torch.Tensor,
                   alpha: float = 1.0):
    """CUDA version of :func:`cutmix3d`."""
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    D, H, W = vol_a.shape
    cut_ratio = (1.0 - lam) ** (1.0 / 3.0)
    dz = max(1, int(D * cut_ratio))
    dy = max(1, int(H * cut_ratio))
    dx = max(1, int(W * cut_ratio))
    z0 = int(torch.randint(0, max(1, D - dz + 1), (1,)))
    y0 = int(torch.randint(0, max(1, H - dy + 1), (1,)))
    x0 = int(torch.randint(0, max(1, W - dx + 1), (1,)))
    z1, y1, x1 = z0 + dz, y0 + dy, x0 + dx
    vol = vol_a.clone()
    vol[z0:z1, y0:y1, x0:x1] = vol_b[z0:z1, y0:y1, x0:x1]
    cls = cls_a.clone()
    off = off_a.clone()
    gz0, gy0, gx0 = z0 // 2, y0 // 2, x0 // 2
    gz1, gy1, gx1 = z1 // 2, y1 // 2, x1 // 2
    cls[:, gz0:gz1, gy0:gy1, gx0:gx1] = cls_b[:, gz0:gz1, gy0:gy1, gx0:gx1]
    off[:, gz0:gz1, gy0:gy1, gx0:gx1] = off_b[:, gz0:gz1, gy0:gy1, gx0:gx1]
    return vol, cls, off

