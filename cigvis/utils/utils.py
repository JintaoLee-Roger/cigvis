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


def check_mmap(d: np.ndarray) -> None:
    if isinstance(d, np.memmap):
        if d.mode != 'r' and d.mode != 'c':
            warnings.warn(
                f"Your memmap data mode is '{d.mode}'. " +
                f"We strongly recommend using `mode='r'` or " +
                f"`mode='c'`, as `mode='{d.mode}'` may change " +
                f"file in some cases", UserWarning)


def deprecated(custom_message=None, replacement=None):
    """Decorator to mark functions as deprecated with an optional custom message
    and replacement function name.

    :param custom_message: (str) Custom deprecation message
    :param replacement: (str) The name of the replacement function
    """

    def decorator(func):

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            message = f"Call to deprecated function {func.__name__}."
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
    v1 = float(nmin(d))
    v2 = float(nmax(d))
    if v1 == v2:
        return [v1 - 0.1, v1 + 0.2]
    if v1 * v2 < 0:
        if abs(v1) / abs(v2) < 0.05 or abs(v1) / abs(v2) > 20:
            return [v1 * scale, v2 * scale]
        else:
            v = min(abs(v1), abs(v2)) * scale
            return [-v, v]
    return [v1 * scale, v2 * scale]
