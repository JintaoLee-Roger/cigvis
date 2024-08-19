# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
surface
"""

from typing import Tuple
import numpy as np


def surface2mesh(surf: np.ndarray,
                 mask: np.ndarray = None,
                 mask_type: str = 'e',
                 anti_rot: bool = True,
                 step1=2,
                 step2=2) -> Tuple:
    n1, n2 = surf.shape
    surf = surf[::step1, ::step2]
    mask = mask[::step1, ::step2] if mask is not None else None
    n1g, n2g = surf.shape

    if mask is not None and mask_type == 'e':
        mask = ~mask

    # set grid
    if mask is None:
        grid = np.arange(n1g * n2g).reshape(n1g, n2g)
    else:
        grid = -np.ones_like(mask, dtype=int)
        grid[mask] = np.arange(mask.sum())

    # get vertices
    y, x = np.meshgrid(np.arange(0, n2, step2), np.arange(0, n1, step1))
    vertices = np.stack((x, y, surf), axis=-1).reshape(-1, 3)
    if mask is not None:
        vertices = vertices[mask.flatten()]

    faces = np.zeros((n1g - 1, n2g - 1, 2, 3))
    faces[:, :, 0, 0] = grid[:-1, :-1]
    faces[:, :, 0, 1] = grid[1:, :-1]
    faces[:, :, 0, 2] = grid[1:, 1:]
    faces[:, :, 1, 0] = grid[:-1, :-1]
    faces[:, :, 1, 1] = grid[1:, 1:]
    faces[:, :, 1, 2] = grid[:-1, 1:]

    faces = faces.reshape(-1, 3).astype(int)
    faces = faces[~np.any(faces == -1, axis=1)]
    if anti_rot:
        faces[:, [1, 2]] = faces[:, [2, 1]]

    return vertices, faces.astype(int)


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
