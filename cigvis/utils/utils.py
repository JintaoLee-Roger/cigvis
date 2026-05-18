# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
some utils
"""

import warnings
import numpy as np
import functools

DEPRECATION_VERSION = '0.2.1'
DEPRECATION_REMOVAL_VERSION = '0.4.0'


def check_mmap(d: np.ndarray) -> None:
    if isinstance(d, dict):
        seen = set()
        for value in d.values():
            if id(value) in seen:
                continue
            seen.add(id(value))
            check_mmap(value)
        return
    if isinstance(d, np.memmap):
        if d.mode != 'r' and d.mode != 'c':
            warnings.warn(
                f"Your memmap data mode is '{d.mode}'. " +
                f"We strongly recommend using `mode='r'` or " +
                f"`mode='c'`, as `mode='{d.mode}'` may change " +
                f"file in some cases", UserWarning)


def deprecated(
    custom_message=None,
    replacement=None,
    deprecated_in=DEPRECATION_VERSION,
    remove_in=DEPRECATION_REMOVAL_VERSION,
):
    """Decorator to mark functions as deprecated with an optional custom message
    and replacement function name.

    :param custom_message: (str) Custom deprecation message
    :param replacement: (str) The name of the replacement function
    :param deprecated_in: (str) Version where the API was deprecated
    :param remove_in: (str) Version where the API is scheduled for removal
    """

    def decorator(func):

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            message = (
                f"Call to deprecated function {func.__name__}. "
                f"Deprecated since {deprecated_in}; scheduled for removal "
                f"in {remove_in}."
            )
            if replacement:
                message += f" Use {replacement} instead."
            if custom_message:
                message += f" {custom_message}"
            warnings.simplefilter('always',
                                  DeprecationWarning)  # turn off filter
            warnings.warn(message, category=DeprecationWarning, stacklevel=2)
            warnings.simplefilter('default',
                                  DeprecationWarning)  # reset filter
            return func(*args, **kwargs)

        return new_func

    return decorator


def mmap_min(d: np.ndarray):
    if isinstance(d, np.memmap):
        if d.ndim < 3:
            return np.nanmin(d)
        else:
            ni = d.shape[0]
            if ni < 10:
                return np.nanmin(d)
            m1 = np.nanmin(d[:5])
            m2 = np.nanmin(d[-5:])
            m3 = np.nanmin(d[ni // 2 - 2:ni // 2 + 3])
            return min([m1, m2, m3])


def mmap_max(d: np.ndarray):
    if isinstance(d, np.memmap):
        if d.ndim < 3:
            return np.nanmax(d)
        else:
            ni = d.shape[0]
            if ni < 10:
                return np.nanmax(d)
            m1 = np.nanmax(d[:5])
            m2 = np.nanmax(d[-5:])
            m3 = np.nanmax(d[ni // 2 - 2:ni // 2 + 3])
            return max([m1, m2, m3])


def _is_memmap_backed(d) -> bool:
    base = getattr(d, 'base', None)
    while base is not None:
        if isinstance(base, np.memmap):
            return True
        base = getattr(base, 'base', None)
    return False


def _is_in_memory_ndarray(d) -> bool:
    return (isinstance(d, np.ndarray) and not isinstance(d, np.memmap)
            and not _is_memmap_backed(d))


def _sample_block_shape(shape, max_items=65536):
    ndim = len(shape)
    if ndim == 0:
        return ()

    edge = max(1, int(max_items**(1 / ndim)))
    return tuple(max(1, min(int(size), edge)) for size in shape)


def _sample_block_slices(shape):
    shape = tuple(int(size) for size in shape)
    if len(shape) == 0:
        return [()]

    block_shape = _sample_block_shape(shape)
    center = tuple((size - block) // 2
                   for size, block in zip(shape, block_shape))

    starts = [center]
    for axis, (size, block) in enumerate(zip(shape, block_shape)):
        for start in (0, max(0, size - block)):
            item = list(center)
            item[axis] = start
            item = tuple(item)
            if item not in starts:
                starts.append(item)

    return [
        tuple(slice(start, start + block)
              for start, block in zip(item, block_shape))
        for item in starts
    ]


def _sample_to_numpy(sample):
    if is_torch_tensor(sample):
        sample = sample.detach().cpu().numpy()
    return np.asarray(sample)


def _sampled_minmax(d):
    shape = getattr(d, 'shape', None)
    if shape is None:
        return nmin(d), nmax(d)

    mins = []
    maxs = []
    for slices in _sample_block_slices(shape):
        sample = d if slices == () else d[slices]
        arr = _sample_to_numpy(sample)
        if arr.size == 0:
            continue
        try:
            valid = ~np.isnan(arr)
        except TypeError:
            arr = arr.astype(float)
            valid = ~np.isnan(arr)
        if not np.any(valid):
            continue
        values = arr[valid]
        mins.append(np.min(values))
        maxs.append(np.max(values))

    if not mins:
        return np.nan, np.nan
    return min(mins), max(maxs)


def is_torch_tensor(d):
    if type(d).__module__ == 'torch' and type(d).__name__ == 'Tensor':
        return True
    return False


def nmin(d):
    if isinstance(d, np.memmap):
        return mmap_min(d)
    else:
        if is_torch_tensor(d):
            ma = d.min().item()
            if np.isnan(ma):
                raise ValueError("The minimum value of the tensor is nan")
            return ma
        return np.nanmin(d)


def nmax(d):
    if isinstance(d, np.memmap):
        return mmap_max(d)
    else:
        if is_torch_tensor(d):
            ma = d.max().item()
            if np.isnan(ma):
                raise ValueError("The maximum value of the tensor is nan")
            return ma
        return np.nanmax(d)


def auto_clim(d, scale=1):
    if isinstance(d, dict):
        from .slice_provider import clim_source
        d = clim_source(d)

    if _is_in_memory_ndarray(d):
        vmin, vmax = nmin(d), nmax(d)
    else:
        vmin, vmax = _sampled_minmax(d)

    v1 = _format(float(vmin))
    v2 = _format(float(vmax))
    if v1 == v2:
        return [v1 - 0.1, v1 + 0.2]
    if v1 * v2 < 0:
        if abs(v1) / abs(v2) < 0.05 or abs(v1) / abs(v2) > 20:
            return [v1 * scale, v2 * scale]
        else:
            v = min(abs(v1), abs(v2)) * scale
            return [-v, v]
    return [v1 * scale, v2 * scale]


def _format(v):
    if abs(v) > 1:
        return round(v, 2)
    else:
        return float(f"{v:.2g}")


def get_shape(vol, line_first):
    if isinstance(vol, dict):
        from .slice_provider import clim_source
        vol = clim_source(vol)

    def _eq_3_or_4(k):
        return k == 3 or k == 4

    shape_attr = getattr(vol, 'shape', None)
    if shape_attr is None:
        raise AttributeError("volume-like input must expose a shape attribute")

    ndim = getattr(vol, 'ndim', len(shape_attr))
    assert _eq_3_or_4(ndim), f"Volume's dims must be 3 or 4 (RGB), but got {ndim}"
    rgb_type = 0
    shape = list(shape_attr)
    if len(shape) == 4: # RGB volumes
        if _eq_3_or_4(shape[-1]):
            shape = shape[:3]
            rgb_type = 1
        elif _eq_3_or_4(shape[0]):
            shape = shape[1:]
            rgb_type = 2
        else:
            raise ValueError(f"Unknow input type (shape={shape}).")

    if not line_first:
        shape = shape[::-1]

    return shape, rgb_type
