# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.

"""
global configs
--------------

Data order: is_line_first

axis system: is x/y/z reversed
"""


# Is the first dimension inline?
# If True, volume.shape is like (ni, nx, nt)
# If False, volume.shape is like (nt, nx, ni)
from typing import Tuple

LINE_FIRST = True

# axis reversed
X_REVERSED = False
Y_REVERSED = True
Z_REVERSED = True


def is_line_first() -> bool:
    global LINE_FIRST
    return LINE_FIRST


def set_order(line_first: bool) -> None:
    global LINE_FIRST
    assert isinstance(line_first, bool)
    LINE_FIRST = line_first


def is_x_reversed() -> bool:
    global X_REVERSED
    return X_REVERSED


def set_x_reversed(reverse: bool) -> None:
    global X_REVERSED
    assert isinstance(reverse, bool)
    X_REVERSED = reverse


def is_y_reversed() -> bool:
    global Y_REVERSED
    return Y_REVERSED


def set_y_reversed(reverse: bool) -> None:
    global Y_REVERSED
    assert isinstance(reverse, bool)
    Y_REVERSED = reverse


def is_z_reversed() -> bool:
    global Z_REVERSED
    return Z_REVERSED


def set_z_reversed(reverse: bool) -> None:
    global Z_REVERSED
    assert isinstance(reverse, bool)
    Z_REVERSED = reverse


def is_axis_reversed() -> Tuple:
    global X_REVERSED
    global Y_REVERSED
    global Z_REVERSED

    return X_REVERSED, Y_REVERSED, Z_REVERSED


def set_axis_reversed(x: bool, y: bool, z: bool) -> None:
    global X_REVERSED
    global Y_REVERSED
    global Z_REVERSED
    assert isinstance(x, bool)
    assert isinstance(y, bool)
    assert isinstance(z, bool)

    X_REVERSED = x
    Y_REVERSED = y
    Z_REVERSED = z