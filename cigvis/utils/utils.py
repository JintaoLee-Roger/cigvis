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
