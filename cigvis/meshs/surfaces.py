# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
surface
"""

import numpy as np


def arbline2mesh(p, n3, anti_rot=True, vstep=1):
    """
    Construct a 3d mesh. The mesh is a surface represented an arbitrary line, 
    where the surface parallel to the z axis.

    Parameters
    ----------
    p : ArrayLike
        shape is (N, 2)
    n3 : int
        the number of points in the z axis
    anti_rot : bool
        if True, will rotate the faces to anti-clockwise
    vstep : int
        the step size in the vertical direction

    Returns
    -------
    vertices : ArrayLike
        shape is (N*n3, 3)
    faces : ArrayLike
        the faces of the mesh
    """
    # sort the points according to (x, y) order
    p = p[np.lexsort(p.T)]

    N = p.shape[0]

    t = np.arange(n3)
    if vstep > 1:
        t = t[::vstep]
        n3 = t.shape[0]

    # construct the vertices
    vertices = np.zeros((N * n3, 3))
    vertices[:, 0] = np.repeat(p[:, 0], n3)
    vertices[:, 1] = np.repeat(p[:, 1], n3)
    vertices[:, 2] = np.tile(t, N)

    # construct the faces
    grid = np.arange(N * n3).reshape(N, n3)
    faces = np.zeros((N - 1, n3 - 1, 2, 3))
    faces[:, :, 0, 0] = grid[:-1, :-1]
    faces[:, :, 0, 1] = grid[1:, :-1]
    faces[:, :, 0, 2] = grid[1:, 1:]
    faces[:, :, 1, 0] = grid[:-1, :-1]
    faces[:, :, 1, 1] = grid[1:, 1:]
    faces[:, :, 1, 2] = grid[:-1, 1:]

    faces = faces.reshape(-1, 3).astype(int)
    if anti_rot:
        faces[:, [1, 2]] = faces[:, [2, 1]]

    return vertices, faces.astype(int)