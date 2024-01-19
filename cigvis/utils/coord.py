# Copyright (c) 2024 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
coordinate transform tools
"""

import numpy as np


def get_transform_metrix(p1, p2):
    """
    Calculate affine transform metrix `H` using LSTSQ, and ignore w.

    p2 = H \cdot p1

    Parameters
    -----------
    p1 : array-like
        points in coordinate 1
    p2 : array-like
        points in coordinate 2

    Returns
    --------
    H : array-like
        shape is (3, 3)
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    assert len(p1) == len(p2)
    N = len(p1)
    assert N >= 3

    A = np.zeros((2 * N, 6), float)
    b = p2.flatten()
    for i in range(N):
        A[2 * i] = [p1[i, 0], p1[i, 1], 1, 0, 0, 0]
        A[2 * i + 1] = [0, 0, 0, p1[i, 0], p1[i, 1], 1]
    H, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    H = np.concatenate([H, [0, 0, 1]], 0)

    return H.reshape(3, 3)


def apply_transform(p, transform_metrix, inv=False):
    """
    apply affine transform to translate the coordinate
    """
    p = np.array(p)
    if p.ndim == 1:
        p = p[np.newaxis, :]
    if inv:
        transform_metrix = np.linalg.inv(transform_metrix)

    p = np.concatenate([p, np.ones((len(p), 1))], 1).T

    out = np.dot(transform_metrix, p)

    x, y, w = out
    x /= w
    y /= w

    return np.c_[x, y]
