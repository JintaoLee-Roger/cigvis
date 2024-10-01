# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2023, modified by Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC)
#
# Copyright (C) 2019 Yunzhi Shi @ The University of Texas at Austin.
# All rights reserved.
# Distributed under the MIT License. See LICENSE for more info.
# -----------------------------------------------------------------------------

from typing import List, Callable, Tuple, Union
import numpy as np
from vispy.color import Colormap
from cigvis import is_line_first
from cigvis.utils import utils


from .axis_aligned_image import AxisAlignedImage, get_image_func

__all__ = ["volume_slices"]


def volume_slices(volumes: Union[np.ndarray, List],
                  x_pos: Union[List, int] = None,
                  y_pos: Union[List, int] = None,
                  z_pos: Union[List, int] = None,
                  preproc_funcs: Callable = None,
                  cmaps: Union[str, Colormap, List] = 'grays',
                  clims: Union[List, Tuple] = None,
                  interpolation: str = 'linear',
                  method: str = 'auto',
                  texture_format=None) -> List[AxisAlignedImage]:
    """ 
    Acquire a list of slices in the form of AxisAlignedImage.
    The list can be attached to a VisCanvas to visualize the volume
    in 3D interactively.

    Parameters
    ----------
    volumes : np.ndarray or List[np.ndarray]
        input 3D volumes
    x_pos : List or int
        x postions
    y_pos : List or int
        y postions
    z_pos : List or int
        z positions
    preproc_funcs : Callable[[np.ndarray], np.ndarray]
        A function to preprocess a slice
    cmaps : List[str or vispy.color.Colormap]
        colormaps
    clims : List or Tuple
        [vmin, vmax] for each volume
    interpolation : str
        interpolation methods for volumes
    method : str
        Selects method of rendering image in case of non-linear transforms., 
        see: https://vispy.org/api/vispy.scene.visuals.html#vispy.scene.visuals.Image

    Returns
    -------
    visual_nodes : List[AxisAlignedImage]
        return seismic_canvas's visual_nodes
    """
    volumes, preproc_funcs, cmaps, clims, interpolation, n_vol, shape = _process_args(
        volumes, preproc_funcs, cmaps, clims, interpolation)

    slices_list = []

    # Function that returns the limitation of slice movement.
    def limit(axis):
        if axis == 'x': return (0, shape[0] - 1)
        elif axis == 'y': return (0, shape[1] - 1)
        elif axis == 'z': return (0, shape[2] - 1)

    # Organize the slice positions.
    for xyz_pos in (x_pos, y_pos, z_pos):
        if not (isinstance(xyz_pos,
                           (list, tuple, int, float)) or xyz_pos is None):
            raise ValueError(
                'Wrong type of x_pos/y_pos/z_pos={}'.format(xyz_pos))
    axis_slices = {'x': x_pos, 'y': y_pos, 'z': z_pos}

    # Create AxisAlignedImage nodes and append to the slices_list.
    for axis, pos_list in axis_slices.items():
        if pos_list is not None:
            if isinstance(pos_list, (int, float)):
                # make it iterable, even only one element
                pos_list = [pos_list]
            for pos in pos_list:
                pos = int(np.round(pos))

                # Generate a list of image funcs for each input volume.
                image_funcs = []
                for i in range(n_vol):
                    image_funcs.append(
                        get_image_func(axis, volumes[i], preproc_funcs[i]))

                # Construct the AxisAlignedImage node.
                image_node = AxisAlignedImage(image_funcs,
                                              axis=axis,
                                              pos=pos,
                                              limit=limit(axis),
                                              cmaps=cmaps,
                                              clims=clims,
                                              interpolation=interpolation,
                                              method=method,
                                              texture_format=texture_format)

                slices_list.append(image_node)

    return slices_list


def _process_args(volumes: Union[np.ndarray, List],
                  preproc_funcs: Callable = None,
                  cmaps: Union[str, Colormap, List] = 'grays',
                  clims: Union[List, Tuple] = None,
                  interpolation: str = 'linear') -> Tuple:
    # check and init
    # Check whether single volume or multiple volumes are provided.
    if isinstance(volumes, (tuple, list)):
        n_vol = len(volumes)
        if preproc_funcs is None:
            preproc_funcs = [None] * n_vol  # repeat n times ...
        else:
            assert isinstance(preproc_funcs, (tuple, list)) \
              and len(preproc_funcs) >= n_vol
        assert isinstance(cmaps, (tuple, list)) \
          and len(cmaps) >= n_vol
        assert isinstance(clims, (tuple, list)) \
          and len(clims) >= n_vol \
          and len(clims[0]) == 2 or clims[0] is None
        for vol in volumes:
            assert vol.shape == volumes[0].shape
    else:
        volumes = [volumes]
        preproc_funcs = [preproc_funcs]
        cmaps = [cmaps]
        clims = [clims]
        n_vol = 1
    if isinstance(interpolation, str):
        interpolation = [interpolation] * len(volumes)
    assert len(interpolation) == len(volumes)

    shape = volumes[0].shape
    line_first = is_line_first()
    if not line_first:
        shape = shape[::-1]

    # Automatically set clim (cmap range) if not specified.
    for i_vol in range(n_vol):
        clim = clims[i_vol]
        vol = volumes[i_vol]
        if clim is None or clim == 'auto':
            if type(vol) == np.memmap:
                from warnings import warn
                warn(
                    "cmap='auto' with np.memmap can significantly impact launching "
                    + "time, clim=(cmin, cmax) is recommended.",
                    UserWarning,
                    stacklevel=2)
            clims[i_vol] = utils.auto_clim(vol)

    return volumes, preproc_funcs, cmaps, clims, interpolation, n_vol, shape
