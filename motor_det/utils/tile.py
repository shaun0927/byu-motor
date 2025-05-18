import math
from typing import Tuple, Iterable, Union

import numpy as np


def as_tuple_of_3(value) -> Tuple:
    if isinstance(value, (int, float)):
        result = value, value, value
    else:
        a, b, c = value
        result = a, b, c

    return result


def compute_better_tiles_1d(length: int, window_size: int, num_tiles: int):
    """Evenly distribute tiles along a 1D axis."""
    last_tile_start = length - window_size

    starts = np.linspace(0, last_tile_start, num_tiles, dtype=int)
    ends = starts + window_size
    for start, end in zip(starts, ends):
        yield slice(start, end)


def compute_better_tiles(
    volume_shape: Tuple[int, int, int],
    window_size: Union[int, Tuple[int, int, int]],
    window_step: Union[int, Tuple[int, int, int]],
) -> Iterable[Tuple[slice, slice, slice]]:
    """Compute the slices for a sliding window over a volume."""
    window_size_z, window_size_y, window_size_x = as_tuple_of_3(window_size)
    window_step_z, window_step_y, window_step_x = as_tuple_of_3(window_step)
    z, y, x = volume_shape

    num_z_tiles = math.ceil(z / window_step_z)
    num_y_tiles = math.ceil(y / window_step_y)
    num_x_tiles = math.ceil(x / window_step_x)

    for z_slice in compute_better_tiles_1d(z, window_size_z, num_z_tiles):
        for y_slice in compute_better_tiles_1d(y, window_size_y, num_y_tiles):
            for x_slice in compute_better_tiles_1d(x, window_size_x, num_x_tiles):
                yield (
                    z_slice,
                    y_slice,
                    x_slice,
                )


def compute_better_tiles_with_num_tiles(
    volume_shape: Tuple[int, int, int],
    window_size: Union[int, Tuple[int, int, int]],
    num_tiles: Tuple[int, int, int],
) -> Iterable[Tuple[slice, slice, slice]]:
    """Compute the slices for a sliding window over a volume using a fixed number of tiles."""
    window_size_z, window_size_y, window_size_x = as_tuple_of_3(window_size)
    num_z_tiles, num_y_tiles, num_x_tiles = as_tuple_of_3(num_tiles)
    z, y, x = volume_shape

    for z_slice in compute_better_tiles_1d(z, window_size_z, num_z_tiles):
        for y_slice in compute_better_tiles_1d(y, window_size_y, num_y_tiles):
            for x_slice in compute_better_tiles_1d(x, window_size_x, num_x_tiles):
                yield (
                    z_slice,
                    y_slice,
                    x_slice,
                )
