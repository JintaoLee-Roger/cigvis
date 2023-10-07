# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Load three types of fault skin. One fault skin file 
contains three parts: header, mid, tail

- classic fault skin: 
    header: 1 int32 number for ncells, 
    mid: 9*ncells float32 numbers, [z, y, x, likelihood, strike, dip, slip[3]]
    tail: 4*ncells int32 numbers, [above, below, left, right] indices

- 12 numbers tail fault skin:
    header: 4 int32 numbers, [ncells, i1seed, 12seed, i3seed]
    mid: 12*ncells float32 numbers, [z, y, x, likelihood, strike, dip, ...]
    tail: 12*ncells int32 numbers, [i1a, i2a, i3a, i1b, i2b, i3b, i1l, i2l, i3l, i1r, i2r, i3r]

- .cig format (fault skin and control points)
    header: 2 int32 numbers, [nctrl, ncells]
    mid1: 6 * nctrl float32 numbers, [cpsz, cpsy, cpsx, cusz, cusy, cusx]
    mid2: 6 * ncells float32 numbers, [z, y, x, likelihood, strike, dip]
    tail: 4 * ncells int32 numbers, [above, below, left, right] indices


see: https://github.com/xinwucwp/osv/blob/f4e2564fc27b9539edc4caff0944b1ddb94997b8/src/osv/FaultSkin.java#L569
for classic fault skin
"""

import numpy as np
import glob
from cigvis.meshs import merge_meshs


def load_skins(filedir: str,
               suffix='*',
               endian: str = '>',
               values_type: str = None):
    """
    load muiltiple skin file and merge all vertices and faces into one

    Parameters
    -----------
    filedir : str
        skin files dir
    endian : str
        endianess, big is '>', little is '<'
    values_type : str
        can be 'likelihood', 'strike', 'dip', ...

    Returns
    --------
    vertices : array-like 
        shape is [N, 3]
    faces: array-like 
        shape is [N2, 3]
    vertex_values : array-like
        shape is [N]
    """
    if filedir[-1] != '/':
        filedir += '/'
    flist = glob.glob(filedir + suffix)
    vertices = []
    faces = []
    values = []
    for f in flist:
        vert, face, v = load_one_skin(f, endian, values_type)
        vertices.append(vert)
        faces.append(face)
        values.append(v)

    vertices, faces = merge_meshs(vertices, faces)
    if values_type is not None:
        values = np.concatenate(values)
    else:
        values = None

    return vertices, faces, values


def load_one_skin(filename: str, endian: str = '>', values_type: str = None):
    """
    load one skin file

    Parameters
    ----------
    filename : str
        skin file name
    endian : str
        endianess, big is '>', little is '<'
    values_type : str
        can be 'likelihood', 'strike', 'dip', ...

    Returns
    --------
    vertices : array-like 
        shape is [N, 3]
    faces: array-like 
        shape is [N2, 3]
    vertex_values : array-like
        shape is [N]
    """
    # load file
    mid, tail = _load_skin(filename, endian)

    # vertices
    vertices = mid[:, [2, 1, 0]]

    # vertex_colors
    vertex_values = None
    if values_type is not None:
        vertex_values = mid[:, idx(values_type)]

    # faces
    if tail.shape[1] == 4:
        cells = _cells4_to_faces(tail)
    else:
        cells = _cells12_to_faces(mid, tail)

    faces = []

    for i in range(len(cells)):
        R = cells[i, 2]
        if R is not None:
            RB = cells[R, 1]
        else:
            RB = None
        B = cells[i, 1]
        if B is not None:
            BR = cells[B, 2]
        else:
            BR = None

        if R is not None and RB is not None:
            faces.append([i, cells[RB, 0], cells[R, 0]])
        elif R is not None and BR is not None:
            faces.append([i, cells[BR, 0], cells[R, 0]])
        if B is not None and RB is not None:
            faces.append([i, cells[B, 0], cells[RB, 0]])
        elif B is not None and BR is not None:
            faces.append([i, cells[B, 0], cells[BR, 0]])

    faces = np.array(faces).astype(int)

    return vertices, faces, vertex_values


def _cells12_to_faces(mid: np.ndarray, tail: np.ndarray):
    """
    new verison cells

    Parameters
    -----------
    mid : array-like
        shape is [ncells, 12], each row is like [z, y, x, ...]
    tail : array-like
        shape is [ncells, 12], each row is like
        [i1a, i2a, i3a, i1b, i2b, i3b, i1l, i2l, i3l, i1r, i2r, i3r],
        i1 means z, i2 means y, i3 means x

    Returns
    --------
    cells : array-like
        shape is [ncells, 3], each row is like 
        [idx, below_cell_index, right_cell_index]
    """
    pos = mid[:, [2, 1, 0]]
    intpos = np.round(pos).astype(np.int32)
    xmin = intpos[:, 0].min()
    ymin = intpos[:, 1].min()
    zmin = intpos[:, 2].min()
    nx = 1 + intpos[:, 0].max() - xmin
    ny = 1 + intpos[:, 1].max() - ymin
    nz = 1 + intpos[:, 2].max() - zmin
    tz = tail[:, ::3] - zmin
    ty = tail[:, 1::3] - ymin
    tx = tail[:, 2::3] - xmin
    tz[np.logical_or(tz < 0, tz > nz)] = nz
    ty[np.logical_or(ty < 0, ty > ny)] = ny
    tx[np.logical_or(tx < 0, tx > nx)] = nx

    ix = intpos[:, 0] - xmin
    iy = intpos[:, 1] - ymin
    iz = intpos[:, 2] - zmin
    cell_idx = np.full((nx + 1, ny + 1, nz + 1), None)
    cell_idx[ix, iy, iz] = np.arange(len(mid))

    # cells = np.full((len(mid), 3), None)
    # for i in range(len(mid)):
    #     b = cell_idx[tx[i, 1], ty[i, 1], tz[i, 1]]
    #     r = cell_idx[tx[i, 3], ty[i, 3], tz[i, 3]]
    #     cells[i] = [i, b, r]

    b_indices = cell_idx[tx[:, 1], ty[:, 1], tz[:, 1]]
    r_indices = cell_idx[tx[:, 3], ty[:, 3], tz[:, 3]]
    cells = np.column_stack((np.arange(len(mid)), b_indices, r_indices))

    return cells


def _cells4_to_faces(tail: np.ndarray):
    """
    old verison cells

    Parameters
    -----------
    tail : array-like
        shape is [ncells, 4], each row is like [above, below, left, right] 
        indices

    Returns
    --------
    cells : array-like
        shape is [ncells, 3], each row is like 
        [idx, below_cell_index, right_cell_index]
    """
    ncell = len(tail)
    tail = tail.astype(object)
    tail[tail < 0] = None
    b_indices = tail[:, 1]
    r_indices = tail[:, 3]
    cells = np.column_stack((np.arange(ncell), b_indices, r_indices))

    return cells


def idx(name: str):
    if name == 'x' or name == 'iline':
        ix = 2
    elif name == 'y' or name == 'xline':
        ix = 1
    elif name == 'z' or name == 'time':
        ix = 0
    elif name == 'likelihood':
        ix = 3
    elif name == 'strike' or name == 'phi':
        ix = 4
    elif name == 'dip' or name == 'theta':
        ix = 5
    elif name == 'vector':
        ix = slice(6, 9)
    else:
        raise ValueError("Unknow key name")

    return ix


def _load_skin(filename: str, endian: str = '>'):
    """
    The structure of skin file: header, mid, tail
    header: ncells, (i1seed, i2seed, i3seed)(new version), int32
    mid: [z, y, x, likelihood, strike, dip, ...] * ncells, float32
    tail: [above, below, left, right] * ncells, int32

    load a skin file into two parts: mid and tail

    Parameters
    -----------
    filename : str
        skin file name
    endian : str
        little endian is '<', big endian is '>'

    Returns
    --------
    mid : array-like
        shape is [ncells, 12] or [ncells, 9] (old version)
    tail : array-like
        shape is [ncells, 12] or [ncells, 4] (old version)
    """
    d = np.fromfile(filename, dtype=np.int32)
    if endian == '>':
        d = d.byteswap()
    if (d[0] * 24 + 4) == len(d) or (d[0] * 13 + 4) == len(d):
        ncell = d[0]

        nele = (len(d) - 4) // ncell
        nattr = 12 if nele == 24 else 9
        nnigb = 12 if nele == 24 else 4

        mid = d[4:4 + ncell * nattr].view(np.float32).reshape(ncell, nattr)
        tail = d[4 + ncell * nattr:].reshape(ncell, nnigb)
    elif (2 + d[0] * 6 + d[1] * 10) == len(d):
        """.cig format"""
        nctrl = d[0]
        ncell = d[1]
        start = 2 + nctrl * 6
        mid = d[start:start + ncell * 6].view(np.float32).reshape(ncell, 6)
        tail = d[start + ncell * 6:].reshape(ncell, 4)
    else:
        raise RuntimeError("Unknow file type")

    return mid, tail
