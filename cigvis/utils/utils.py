# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.

"""
some utils
"""

import warnings
import numpy as np


def check_mmap(d: np.ndarray) -> None:
    if isinstance(d, np.memmap):
        if d.mode != 'r' and d.mode != 'c':
            warnings.warn(
                f"Your memmap data mode is '{d.mode}'. " +
                f"We strongly recommend using `mode='r'` or " +
                f"`mode='c'`, as `mode='{d.mode}'` may change " +
                f"file in some cases", UserWarning)
