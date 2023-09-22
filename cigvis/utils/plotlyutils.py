# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.

"""
utils for plotly visualization in jupyter

TODO: To be improved
"""


from typing import Union, Tuple, List, Dict
import numpy as np

import cigvis


def make_xyz(idx: int, shape: Union[Tuple, List],
             axis: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    make xx, yy, zz (meshgrid) for plotly

    Parameters
    ----------
    idx : int
        index
    shape : List or Tuple
        len(shape) == 3, data's shape
    axis : str
        axis can be: 'x' or 'inline', 'y' or 'crossline', 'z' or 'time'

    Returns
    -------
    xx : np.ndarray
    yy : np.ndarray
    zz : np.ndarray

    """
    line_first = cigvis.is_line_first()

    if line_first:
        ni, nx, nt = shape
    else:
        nt, nx, ni = shape

    if axis == 'x' or axis == 'inline':
        assert idx >= 0 and idx < ni
        yy, zz = np.meshgrid(np.arange(nx), np.arange(nt))
        xx = idx * np.ones((nt, nx))
    elif axis == 'y' or axis == 'crossline':
        assert idx >= 0 and idx < nx
        xx, zz = np.meshgrid(np.arange(ni), np.arange(nt))
        yy = idx * np.ones((nt, ni))
    elif axis == 'z' or axis == 'time':
        assert idx >= 0 and idx < nt
        xx, yy = np.meshgrid(np.arange(ni), np.arange(nx))
        zz = idx * np.ones((nx, ni))
    else:
        raise ValueError(f"Invalid value of axis: {axis}")

    return xx, yy, zz


def get_image_func(volume, axis, idx, prefunc=None):
    """
    get a slice image from a volume with axis and idx
    """
    line_first = cigvis.is_line_first()

    if line_first:
        if axis == 'x':
            out = volume[idx, :, :].T
        elif axis == 'y':
            out = volume[:, idx, :].T
        elif axis == 'z':
            out = volume[:, :, idx].T
    else:
        if axis == 'x':
            out = volume[:, :, idx]
        elif axis == 'y':
            out = volume[:, idx, :]
        elif axis == 'z':
            out = volume[idx, :, :]

    if prefunc is not None:
        out = prefunc(out)

    return out


def make_slices(data: np.ndarray,
                x: List or int = [],
                y: List or int = [],
                z: List or int = [],
                pos: Dict = None) -> Tuple[Dict, Dict]:
    """
    make slices and locations for plotly

    Parameters
    ----------
    data : np.ndarray
        Input data
    x : List or int
        x or inline index
    y : List or int
        y or crossline index
    z : List or int
        z or time index

    Returns
    -------
    slices : Dict
        slices, Dict[str: np.ndarray]
    pos : Dict
        positions of the slices
    """

    slices = {'x': [], 'y': [], 'z': []}
    if pos is None:
        x = [x] if isinstance(x, int) else x
        y = [y] if isinstance(y, int) else y
        z = [z] if isinstance(z, int) else z

        pos = {'x': x, 'y': y, 'z': z}

    assert isinstance(pos, Dict)

    for axis in pos.keys():
        for idx in pos[axis]:
            s = get_image_func(data, axis, idx)
            slices[axis].append(s)

    return slices, pos


def verifyshape(sshape: tuple,
                shape: tuple,
                axis: str) -> None:
    """
    verify the slice shape is invalid
    Note: sshape is slice.shape which is transposed if line_first is False
    """
    line_first = cigvis.is_line_first()

    if line_first:
        ni, nx, nt = shape
    else:
        nt, nx, ni = shape
    if axis == 'x' or axis == 'inline':
        assert sshape == (nt, nx)
    elif axis == 'y' or axis == 'crossline':
        assert sshape == (nt, ni)
    elif axis == 'z' or axis == 'time':
        assert sshape == (nx, ni)
    else:
        raise ValueError(f"Invalid value of axis: {axis}")


######## **kwargs for figure *******************
def kwargs_todict(v):
    if isinstance(v, dict):
        return v
    elif isinstance(v, (int, float, bool, str)):
        return dict(x=v, y=v, z=v)
    elif isinstance(v, list) or isinstance(v, tuple):
        assert len(v) == 3
        return dict(x=v[0], y=v[1], z=v[2])
    else:
        raise TypeError(f"Invalid type of v: {type(v)}")


def kwargs_toaxies(v, key, scene):
    out = []
    if isinstance(v, (int, float, bool, str)):
        out = [v, v, v]
    elif isinstance(v, (list, tuple)):
        out = v
    elif isinstance(v, dict):
        out = [v['x'], v['y'], v['z']]
    else:
        raise TypeError(f"Invalid type of v: {type(v)}")

    scene['xaxis'][key] = out[0]
    scene['yaxis'][key] = out[1]
    scene['zaxis'][key] = out[2]
    return scene


def make_3Dscene(**kwargs):
    scene = {'camera': {}, 'xaxis': {}, 'yaxis': {}, 'zaxis': {}}

    # modify eye to change view
    # x < 0, clockwise, z > 0, top -> down
    if cigvis.is_z_reversed() and cigvis.is_y_reversed():
        scene['camera']['eye'] = kwargs_todict(
            kwargs.get('eye', dict(x=1.25, y=-1.5, z=1.5)))
    elif cigvis.is_z_reversed() and not cigvis.is_y_reversed():
        scene['camera']['eye'] = kwargs_todict(
            kwargs.get('eye', dict(x=1.5, y=1.5, z=1.5)))

    scene['camera']['center'] = kwargs_todict(kwargs.get('center', 0))
    scene['camera']['up'] = kwargs_todict(kwargs.get('up', dict(x=0, y=0,
                                                                z=1)))
    scene['aspectratio'] = kwargs.get('aspectratio', None)
    scene['aspectmode'] = kwargs.get('aspectmode', 'data')

    if cigvis.is_x_reversed():
        scene['xaxis']['autorange'] = 'reversed'
    if cigvis.is_y_reversed():
        scene['yaxis']['autorange'] = 'reversed'
    if cigvis.is_z_reversed():
        scene['zaxis']['autorange'] = 'reversed'

    if 'clean' in kwargs and kwargs.get('clean'):
        scene = kwargs_toaxies(False, 'showticklabels', scene)
        scene = kwargs_toaxies(False, 'showbackground', scene)
        scene = kwargs_toaxies(False, 'showspikes', scene)
        scene = kwargs_toaxies('', 'title', scene)
        return scene

    scene = kwargs_toaxies(kwargs.get('nticks', 5), 'nticks', scene)

    optionkeys = [
        'showticklabels', 'showbackground', 'showspikes', 'backgroundcolor',
        'linecolor', 'gridcolor', 'spikecolor'
    ]

    for key in optionkeys:
        if key in kwargs:
            scene = kwargs_toaxies(kwargs.get(key), key, scene)
    if 'tickcolor' in kwargs:
        scene = kwargs_toaxies(kwargs.get('tickcolor'), 'color', scene)

    if 'showtitle' in kwargs and not kwargs.get('showtitle'):
        title = ['', '', '']
    else:
        title = ['Inline', 'Crossline', 'Time']
    scene['xaxis']['title'] = title[0]
    scene['yaxis']['title'] = title[1]
    scene['zaxis']['title'] = title[2]

    return scene
