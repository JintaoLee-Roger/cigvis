# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
utils to process surface, such as create triangular mesh
"""

from typing import List, Union, Tuple
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
from cigvis import is_line_first

##### Triangular meshing #######


def get_vertices_and_faces(surf: np.ndarray,
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


def fill_grid(points: np.ndarray,
              shape: Union[List, Tuple],
              interp: bool = True,
              method: str = 'cubic',
              fill: float = -1) -> np.ndarray:
    """
    fill `points` in a grid in `shape`. If interp, interpolation,
    this function doesn't extrapolate, so we fill the edge
    as `fill`.

    Parameters
    ----------
    points : array-like
        shape as (N, 3), N points, each row means (x, y, z) coordinate
    shape : List or Tuple
        [n1, n2], the grid shape, points will be filled into a n1*n2 grid
    interp : bool
        if True, will interpolate the grid from points, but not extrapolate,
        this means some edge points will be `fill`
    method : str
        interpolation method
    fill : float
        the points can not be interpolated will be filled as `fill`, default is -1 
    """

    n1, n2 = shape
    assert points[:, 0].max() < n1 and points[:, 0].min() >= 0
    assert points[:, 1].max() < n2 and points[:, 1].min() >= 0

    if interp:
        coords = points[:, :2]
        values = points[:, 2]
        y, x = np.meshgrid(np.arange(n2), np.arange(n1))

        grid = griddata(coords, values, (x, y), method=method)
        grid[np.isnan(grid)] = fill
    else:
        n3 = points.shape[1] - 2
        shape = (n1, n2, n3)
        grid = np.full(shape, fill).astype(float)
        x = points[:, 0].astype(int)
        y = points[:, 1].astype(int)
        grid[x, y, ...] = points[:, 2:]
        if n3 == 1:
            grid = grid.reshape(n1, n2)

    return grid


##### preprocess ######


def preproc_surf_array2(surf, volume=None, value_type='depth'):
    """
    surf shape as (n1, n2)
    """
    assert surf.ndim == 2
    if value_type == 'depth':
        value = None
    else:
        assert volume is not None
        value = interp_surf(volume, surf)

    return surf, value, None


def preproc_surf_array3(surf, value_type='depth'):
    """
    surf shape as (n1, n2, 2) or (n1, n2, 4) or (n1, n2, 5)
    """
    assert surf.ndim == 3
    if value_type == 'depth':
        value = None
        color = None
    else:
        if surf.shape[2] == 2:
            value = surf[:, :, 1]
            color = None
        elif surf.shape[2] == 4 or surf.shape[2] == 5:
            value = None
            color = surf[:, :, 1:]
        else:
            raise RuntimeError("Invalid shape")

    return surf[:, :, 0], value, color


def preproc_surf_pos(surf,
                     shape,
                     volume=None,
                     value_type='depth',
                     interp=True,
                     method='cubic',
                     fill=-1):
    """
    surf shape as (N, 3) or (N, 4) or (N, 6) or (N, 7)
    """
    assert surf.ndim == 2
    assert surf.shape[1] >= 3

    out_surf = fill_grid(surf[:, :3], shape, interp, method, fill)

    if value_type == 'depth':
        value = None
        color = None
    else:
        if surf.shape[1] == 3:
            assert volume is not None
            value = interp_surf(volume, out_surf)
            color = None
        elif surf.shape[1] == 4:
            value = surf[:, 3]
            # fill grid
            value = fill_grid(surf[:, [0, 1, 3]], shape, interp, method,
                              np.nan)
            color = None
        elif surf.shape[1] == 6 or surf.shape[1] == 7:
            assert interp == False
            value = None
            indices = [0, 1, 3, 4, 5]
            if surf.shape[1] == 7:
                indices += [6]
            color = fill_grid(surf[:, indices], shape, False, fill=np.nan)
        else:
            raise RuntimeError("Invalid shape")

    return out_surf, value, color


##### interpolation #####


def interp_linefirst_impl(volume: np.ndarray, surf: np.ndarray,
                          order: int) -> np.ndarray:
    """
    interp value of linefirst volume in surf positions

    Parameters
    ----------
    volume : array-like
        3D array, shape as (ni, nx, nt)
    surf : array-like
        2D array, shape as (ni, nx), each point means z pos

    Returns
    -------
    value : array-like
        2D array, shape as (ni, nx), each point means value in z pos
    """
    ni, nx, nt = volume.shape
    x, y = np.meshgrid(np.arange(ni), np.arange(nx), indexing='ij')
    coordinates = np.vstack((x.ravel(), y.ravel(), surf.ravel()))
    out = map_coordinates(volume, coordinates, order=order, mode='reflect')
    return out.reshape(ni, nx)


# @numba.jit(nopython=True)
def interp_timefirst_impl(volume: np.ndarray, surf: np.ndarray,
                          order: int) -> np.ndarray:
    """
    interp value of timefirst volume in surf positions

    Parameters
    ----------
    volume : array-like
        3D array, shape as (nt, nx, ni)
    surf : array-like
        2D array, shape as (nx, ni), each point means z/nt pos

    Returns
    -------
    value : array-like
        2D array, shape as (nx, ni), each point means value in z/nt pos
    """
    nt, nx, ni = volume.shape
    y, x = np.meshgrid(np.arange(nx), np.arange(ni), indexing='ij')
    coordinates = np.vstack((surf.ravel(), y.ravel(), x.ravel()))
    out = map_coordinates(volume, coordinates, order=order, mode='reflect')
    return out.reshape(nx, ni)


def interp_surf(volume: np.ndarray,
                surf: np.ndarray,
                order: int = 1) -> np.ndarray:
    """
    interp value of volume in surf positions

    Parameters
    ----------
    volume : array-like
        3D array
    surf : array-like
        2D array, each point means z/nt pos
    order : int
        interpolate method, 0 for nearest, 1 for linear
    

    Returns
    -------
    value : array-like
        2D array, each point means value in z/nt pos

    """
    line_first = is_line_first()

    if line_first:
        assert volume.shape[:2] == surf.shape
    else:
        assert volume.shape[1:] == surf.shape

    out = np.zeros_like(surf)
    if line_first:
        out = interp_linefirst_impl(volume, surf, order=order)
    else:
        out = interp_timefirst_impl(volume, surf, order=order)

    return out

