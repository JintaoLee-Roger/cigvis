# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
"""

import numpy as np


def cube_points(points, radius):
    """
    """
    assert points.ndim == 2
    assert points.shape[1] == 3
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    p1 = np.c_[x - radius, y - radius, z - radius]
    p2 = np.c_[x - radius, y - radius, z + radius]
    p3 = np.c_[x - radius, y + radius, z - radius]
    p4 = np.c_[x - radius, y + radius, z + radius]
    p5 = np.c_[x + radius, y - radius, z - radius]
    p6 = np.c_[x + radius, y - radius, z + radius]
    p7 = np.c_[x + radius, y + radius, z - radius]
    p8 = np.c_[x + radius, y + radius, z + radius]

    # N, 8, 3
    p = np.stack([p1, p2, p3, p4, p5, p6, p7, p8]).transpose(1, 0, 2)

    # start vert for each point mesh
    offsets = np.arange(len(points)) * 8

    # faces for one points
    vertex_indices = np.array([[0, 3, 2], [0, 1, 3], [0, 6, 2], [0, 4, 6],
                               [0, 5, 1], [0, 4, 5], [4, 7, 6], [4, 5, 7],
                               [2, 7, 3], [2, 6, 7], [1, 7, 3], [1, 5, 7]])
    vertex_indices = vertex_indices.reshape(1, -1)

    # faces.shape = (N, 36)
    faces = vertex_indices + offsets[:, np.newaxis]
    faces = faces.reshape(len(points) * 12, 3)

    return p.reshape(-1, 3).astype(np.float32), faces.astype(np.int32)


def regular_poly_points(points, radius, poly_points):
    """"""
    angles = np.arange(poly_points)