# motor_det/utils/augment.py
import numpy as np


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

    ctr = center.astype(float)
    if jitter:
        ctr += np.random.randint(-jitter, jitter + 1, size=3)

    z0 = int(np.clip(ctr[2] - D // 2, 0, max(1, Z - D)))
    y0 = int(np.clip(ctr[1] - H // 2, 0, max(1, Y - H)))
    x0 = int(np.clip(ctr[0] - W // 2, 0, max(1, X - W)))
    z1, y1, x1 = z0 + D, y0 + H, x0 + W

    patch = np.asarray(volume[z0:z1, y0:y1, x0:x1], dtype=np.uint8)
    return patch, (z0, y0, x0)
