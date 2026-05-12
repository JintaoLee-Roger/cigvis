# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Functions for drawing 3D seismic figure using vispy
----------------------------------------------------


Note
----
Running **not** in jupyter environment

In plotly, for a seismic volume,
- x means inline order
- y means crossline order
- z means time order

- ni means the dimension size of inline / x
- nx means the dimension size of crossline / y
- nt means the dimension size of time / depth / z

"""

from itertools import product
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable, List, Tuple, Dict, Union
import warnings
import os
import numpy as np
from cigvis.vispynodes import (
    VisCanvas,
    volume_slices,
    AxisAlignedImage,
    InteractiveLine,
    Colorbar,
    WellLog,
    XYZAxis,
    SurfaceNode,
    ArbLineNode,
    Axis3D,
    NorthPointer,
)
from cigvis.vispynodes.shading_filter import HeadlightShadingFilter

from vispy.scene.visuals import Mesh, Line
import vispy
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

import cigvis
from cigvis import colormap
from cigvis.utils import surfaceutils
import cigvis.utils as utils
from cigvis.meshs import surface2mesh
from cigvis.vispynodes.screenshot import _normalize_render_size, _save_canvas_png
from cigvis.vispynodes.volume_image import VolumeImage

__all__ = [
    "create_slices",
    "add_mask",
    "create_overlay",
    "create_colorbar",
    "create_colorbar_from_nodes",
    "create_surfaces",
    "set_surface_color_by_slices_nodes",
    "create_bodies",
    "create_bodys",
    "create_line_logs",
    "create_Line_logs",
    "create_well_logs",
    "create_points",
    "create_fault_skin",
    "create_arbitrary_line",
    "create_axis",
    "create_volume_image",
    "Plot3DView",
    "Plot3DSave",
    "Plot3DColorbar",
    "Plot3DGui",
    "plot3D",
    "run",
]


@dataclass
class Plot3DView:
    """Options that control the 3D view/canvas created by ``plot3D``."""

    size: Tuple[int, int] = (800, 600)
    show: bool = True
    grid: Tuple[int, int] = None
    share: bool = False
    xyz_axis: bool = False
    cbar_region_ratio: float = 0.125
    bgcolor: Any = None
    scale_factor: float = None
    center: Any = None
    fov: float = None
    azimuth: float = None
    elevation: float = None
    zoom_factor: float = None
    axis_scales: Tuple[float, float, float] = None
    title: str = None
    keys: str = None


@dataclass
class Plot3DSave:
    """Options that control screenshot/export behavior in ``plot3D``."""

    path: Any = None
    directory: Any = None
    size: Union[Tuple[int, int], str] = None
    mode: str = 'offscreen'
    output_policy: str = 'fit'
    transparent_bg: bool = False
    pad_color: Any = 'white'
    bgcolor: Any = None


@dataclass
class Plot3DColorbar:
    """Options used when updating/saving colorbars in ``plot3D``."""

    save: bool = False
    name: str = 'cbar.png'
    cmap: Any = None
    clim: Any = None
    discrete: bool = None
    disc_ticks: Any = None
    dpi_scale: float = None
    label_str: str = None
    label_color: Any = None
    label_size: Any = None
    tick_size: Any = None
    border_width: Any = None
    border_color: Any = None


@dataclass
class Plot3DGui:
    """Options that control the optional PySide6 GUI shell in ``plot3D``."""

    enabled: bool = True
    theme: str = 'dark'


def _set_visual_metadata(visual, **metadata) -> None:
    did_unfreeze = False
    if hasattr(visual, 'unfreeze'):
        try:
            visual.unfreeze()
            did_unfreeze = True
        except Exception:
            did_unfreeze = False
    try:
        for key, value in metadata.items():
            setattr(visual, key, value)
    finally:
        if did_unfreeze and hasattr(visual, 'freeze'):
            try:
                visual.freeze()
            except Exception:
                pass


def create_slices(volume: np.ndarray,
                  pos: Union[List, Dict] = None,
                  clim: List = None,
                  cmap: str = 'Petrel',
                  interpolation: str = 'cubic',
                  texture_format=None,
                  intersection_lines: bool = True,
                  line_color=(1, 1, 1),
                  line_width=2.0,
                  **kwargs) -> List:
    """
    create a slice node

    Parameters
    ----------
    volume : array-like
        3D array
    pos : List or Dict
        init position of the slices, can be a List or Dict, such as:
        ```
        pos = [0, 0, 200] # x: 0, y: 0, z: 200
        pos = [[0, 200], [9], []] # x: 0 and 200, y: 9, z: None
        pos = {'x': [0, 200], 'y': [1], z: []}
        ```
    clim : List
        [vmin, vmax] for plotting 
    cmap : str or Colormap
        colormap, it can be str or matplotlib's Colormap or vispy's Colormap
    interpolation : str
        interpolation method. If the values of the volume is discrete, we recommand 
        set as 'nearest'
    texture_format : None or 'auto',
        if use None, the NaNs will be clip to clim[1],
        and if use 'auto', the NaNs will be discarded, i.e., transparent
    
    line_color : Tuple
        color for intersection lines and border lines, default is white
    line_width : float
        width for intersection lines and border lines, default is 2.0

    kwargs : Dict
        other kwargs for `volume_slices`

    Returns
    -------
    slices_nodes : List
        list of slice nodes
    """
    utils.check_mmap(volume)
    line_first = cigvis.is_line_first()
    shape, _ = utils.get_shape(volume, line_first)
    nt = shape[2]

    # set pos
    if pos is None:
        pos = dict(x=[0], y=[0], z=[nt - 1])
    if isinstance(pos, List):
        assert len(pos) == 3
        if isinstance(pos[0], List):
            x, y, z = pos
        else:
            x, y, z = [pos[0]], [pos[1]], [pos[2]]
        pos = {'x': x, 'y': y, 'z': z}
    assert isinstance(pos, Dict)

    cmap_name = cmap if isinstance(cmap, str) else getattr(cmap, 'name', None)
    if clim is None:
        clim = utils.auto_clim(volume)
    cmap = colormap.cmap_to_vispy(cmap)

    image_dict = volume_slices(volume,
                               pos['x'],
                               pos['y'],
                               pos['z'],
                               cmaps=cmap,
                               clims=clim,
                               interpolation=interpolation,
                               texture_format=texture_format)

    image_nodes = []
    image_nodes += image_dict['x']
    image_nodes += image_dict['y']
    image_nodes += image_dict['z']
    for image_node in image_nodes:
        _set_visual_metadata(
            image_node,
            _cigvis_cmap_name=cmap_name,
            _cigvis_interpolation=interpolation,
        )
        for image in getattr(image_node, 'overlaid_images', [image_node]):
            _set_visual_metadata(
                image,
                _cigvis_cmap_name=cmap_name,
                _cigvis_interpolation=interpolation,
            )

    lines_nodes = []

    if not intersection_lines:
        return image_nodes

    # X-Y intersection lines
    for x_img, y_img in product(image_dict['x'], image_dict['y']):
        line = InteractiveLine(
            ('x', 'y'),
            shape,
            color=line_color,
            width=line_width,
            antialias=True,
        )
        line.link_image(x_img)
        line.link_image(y_img)
        line.refresh()
        lines_nodes.append(line)

    # X-Z intersection lines
    for x_img, z_img in product(image_dict['x'], image_dict['z']):
        line = InteractiveLine(
            ('x', 'z'),
            shape,
            color=line_color,
            width=line_width,
            antialias=True,
        )
        line.link_image(x_img)
        line.link_image(z_img)
        line.refresh()
        lines_nodes.append(line)

    # Y-Z intersection lines
    for y_img, z_img in product(image_dict['y'], image_dict['z']):
        line = InteractiveLine(
            ('y', 'z'),
            shape,
            color=line_color,
            width=line_width,
            antialias=True,
        )
        line.link_image(y_img)
        line.link_image(z_img)
        line.refresh()
        lines_nodes.append(line)

    # contour lines for X Images
    for x_img in image_dict['x']:
        line = InteractiveLine(
            ('x', ),
            shape,
            color=line_color,
            width=line_width,
            antialias=True,
        )
        line.link_image(x_img)
        line.refresh()
        lines_nodes.append(line)

    # contour lines for Y Images
    for y_img in image_dict['y']:
        line = InteractiveLine(
            ('y', ),
            shape,
            color=line_color,
            width=line_width,
            antialias=True,
        )
        line.link_image(y_img)
        line.refresh()
        lines_nodes.append(line)

    # contour lines for Z Images
    for z_img in image_dict['z']:
        line = InteractiveLine(
            ('z', ),
            shape,
            color=line_color,
            width=line_width,
            antialias=True,
        )
        line.link_image(z_img)
        line.refresh()
        lines_nodes.append(line)

    return image_nodes + lines_nodes


def add_mask(nodes: List,
             volumes: Union[List, np.ndarray],
             clims: Union[List, Tuple] = None,
             cmaps: Union[str, List] = None,
             interpolation: str = 'linear',
             alpha = None,
             excpt = None,
             method: str = 'auto',
             texture_format: str = 'auto',
             preproc_funcs: Callable = None,
             **kwargs) -> List:
    """
    Add Mask/Overlay volumes
    
    Parameters
    -----------
    nodes: List[Node]
        A List that contains `AxisAlignedImage` (may be created by `create_slices`)
    volumes : array-like or List
        3D array(s), foreground volume(s)/mask(s)
    clims : List
        [vmin, vmax] for foreground slices plotting
    cmaps : str or Colormap
        colormap for foreground slices, it can be str or matplotlib's Colormap or vispy's Colormap
    interpolation : str
        interpolation method. If the values of the slices is discrete, we recommand 
        set as 'nearest'
    alpha : float or List[float]
        if alpha is not None, using `colormap.fast_set_cmap` to set cmap
    excpt : None or str
        it could be one of [None, 'min', 'max', 'ramp']

    Returns
    -------
    slices_nodes : List
        list of slice nodes
    """

    if not isinstance(volumes, List):
        volumes = [volumes]

    for volume in volumes:
        # TODO: check shape as same as base image
        utils.check_mmap(volume)

    if clims is None:
        clims = [utils.auto_clim(v) for v in volumes]
    if not isinstance(clims[0], (List, Tuple)):
        clims = [clims]

    if cmaps is None:
        raise ValueError("'cmaps' cannot be 'None'")
    if not isinstance(cmaps, List):
        cmaps = [cmaps] * len(volumes)
    cmap_names = [
        cmap if isinstance(cmap, str) else getattr(cmap, 'name', None)
        for cmap in cmaps
    ]
    if not isinstance(alpha, List):
        alpha = [alpha] * len(volumes)
    if not isinstance(excpt, List):
        excpt = [excpt] * len(volume)
    for i in range(len(cmaps)):
        if alpha[i] is not None:
            cmaps[i] = colormap.fast_set_cmap(cmaps[i], alpha[i], excpt[i])
        cmaps[i] = colormap.cmap_to_vispy(cmaps[i])

    if isinstance(interpolation, str):
        interpolation = [interpolation] * len(volumes)
    if not isinstance(preproc_funcs, List):
        preproc_funcs = [preproc_funcs] * len(volumes)

    for node in nodes:
        if not isinstance(node, AxisAlignedImage):
            continue
        for i in range(len(volumes)):
            node.add_mask(
                volumes[i],
                cmaps[i],
                clims[i],
                interpolation[i],
                method=method,
                texture_format=texture_format,
                preproc_f=preproc_funcs[i],
            )
            image = node.overlaid_images[-1]
            _set_visual_metadata(
                image,
                _cigvis_cmap_name=cmap_names[i],
                _cigvis_interpolation=interpolation[i],
            )

    return nodes


def create_volume_image(
    volume: np.ndarray,
    pos: Union[List, Dict] = None,
    clim: List = None,
    cmap: str = 'Petrel',
    interpolation: str = 'linear',
    **kwargs,
) -> VolumeImage:
    """
    Create a VolumeImage manager for one base volume.

    VolumeImage is preferred over create_slices when the volume data may be
    replaced dynamically (e.g., AI workflow: image -> model -> new image).
    Use replace_overlay_volume() + refresh_overlay() to update overlays without
    rebuilding the scene.

    Parameters
    ----------
    volume : np.ndarray
        3D array
    pos : List or Dict, optional
        Slice positions, same format as create_slices. Default: x=0, y=0, z=-1
    clim : List, optional
        [vmin, vmax]. Default: auto from data percentile
    cmap : str
        Colormap name
    interpolation : str
        Interpolation method

    Returns
    -------
    vi : VolumeImage
        Call vi.nodes() to get the list of nodes for plot3D
    """
    if clim is None:
        clim = utils.auto_clim(volume)

    vi = VolumeImage(
        volume,
        cmap=cmap,
        clim=clim,
        interpolation=interpolation,
        **kwargs,
    )

    if pos is None:
        shape, _ = utils.get_shape(volume, cigvis.is_line_first())
        pos = dict(x=[0], y=[0], z=[shape[2] - 1])
    if isinstance(pos, list):
        assert len(pos) == 3
        if isinstance(pos[0], list):
            x, y, z = pos
        else:
            x, y, z = [pos[0]], [pos[1]], [pos[2]]
        pos = {'x': x, 'y': y, 'z': z}

    vi.create_slices(
        x_pos=pos.get('x'),
        y_pos=pos.get('y'),
        z_pos=pos.get('z'),
    )
    return vi


@utils.deprecated(
    "This compatibility wrapper keeps the old vispy overlay API working.",
    "`cigvis.create_slices` plus `cigvis.add_mask`",
)
def create_overlay(bg_volume: np.ndarray,
                   fg_volume: np.ndarray,
                   pos: Union[List, Dict] = None,
                   bg_clim: List = None,
                   fg_clim: List = None,
                   bg_cmap: str = 'Petrel',
                   fg_cmap: str = None,
                   bg_interpolation: str = 'cubic',
                   fg_interpolation: str = 'cubic',
                   return_cbar: bool = False,
                   cbar_type: str = 'fg',
                   **kwargs) -> List:
    """
    Deprecated compatibility wrapper for the old overlay-slice API.

    Prefer ``create_slices(bg_volume)`` followed by ``add_mask(...)`` for new
    code. This function intentionally preserves the old return shape.
    """
    utils.check_mmap(bg_volume)
    if not isinstance(fg_volume, list):
        fg_volume = [fg_volume]
    for volume in fg_volume:
        assert bg_volume.shape == volume.shape
        utils.check_mmap(volume)

    line_first = cigvis.is_line_first()
    if line_first:
        nt = bg_volume.shape[2]
    else:
        nt = bg_volume.shape[0]

    if pos is None:
        pos = dict(x=[0], y=[0], z=[nt - 1])
    if isinstance(pos, list):
        assert len(pos) == 3
        if isinstance(pos[0], list):
            x, y, z = pos
        else:
            x, y, z = [pos[0]], [pos[1]], [pos[2]]
        pos = {'x': x, 'y': y, 'z': z}
    assert isinstance(pos, dict)

    if bg_clim is None:
        bg_clim = utils.auto_clim(bg_volume)
    if fg_clim is None:
        fg_clim = [utils.auto_clim(v) for v in fg_volume]
    if not isinstance(fg_clim[0], (list, tuple)):
        fg_clim = [fg_clim]

    bg_cmap = colormap.cmap_to_vispy(bg_cmap)
    if fg_cmap is None:
        raise ValueError("'fg_cmap' cannot be 'None'")
    if not isinstance(fg_cmap, list):
        fg_cmap = [fg_cmap] * len(fg_volume)
    for i in range(len(fg_cmap)):
        fg_cmap[i] = colormap.cmap_to_vispy(fg_cmap[i])

    if isinstance(fg_interpolation, str):
        fg_interpolation = [fg_interpolation] * len(fg_volume)

    nodes_dict = volume_slices(
        [bg_volume, *fg_volume],
        pos['x'],
        pos['y'],
        pos['z'],
        cmaps=[bg_cmap, *fg_cmap],
        clims=[bg_clim, *fg_clim],
        interpolation=[bg_interpolation, *fg_interpolation],
    )

    nodes = []
    nodes += nodes_dict['x']
    nodes += nodes_dict['y']
    nodes += nodes_dict['z']

    if return_cbar:
        if cbar_type == 'fg':
            cmap = fg_cmap
            clim = fg_clim
        else:
            cmap = bg_cmap
            clim = bg_clim

        if kwargs.get('discrete', False) and kwargs.get('disc_ticks', None) is None:
            kwargs['disc_ticks'] = [np.unique(fg_volume[0])]

        cbar = create_colorbar(cmap, clim, **kwargs)
        return nodes, cbar

    return nodes


def create_colorbar(cmap,
                    clim: List,
                    discrete: bool = False,
                    disc_ticks: Union[List, Dict] = None,
                    label_str: str = '',
                    **kwargs) -> Colorbar:
    """
    create a `Colorbar` instance. To draw colorbar, must spacify 
    `size` params or call `colorbar.update_size(size)` function.

    Parameters
    ---------- 
    cmap : str
        colormap
    clim : List
        [vmin, vmax] to norm
    discrete : bool
        draw a discrete colorbar or not
    disc_ticks : List or Dict
        contains 2 elements, [values, ticklabels] or 
        {'value': values, 'labels': labels}. values are used to get colors 
        from cmap, ticklabels are the labels of colors
    label_str : str
        colorbar label

    kwargs : Dict
        params for Colorbar
    """
    if cmap is None or clim is None:
        return None
    if isinstance(cmap, str) and cmap in colormap.list_custom_cmap():
        cmap = colormap.get_custom_cmap(cmap)

    cbar = Colorbar(cmap=cmap,
                    clim=clim,
                    discrete=discrete,
                    disc_ticks=disc_ticks,
                    label_str=label_str,
                    **kwargs)

    return cbar


def create_colorbar_from_nodes(nodes,
                               label_str='',
                               select='auto',
                               idx=0,
                               idx2=0,
                               **kwargs):
    """
    nodes : List
        List of nodes
    select : str
        One of 'auto', 'last', 'slices', 'mask', 'surface', 'logs', 'fault_skin', 'line_logs'.
        If 'auto', select 'mask' > 'surface' > 'slices' > 'logs' > 'line_logs' > 'mesh'.
        If 'last', select the node[-1]
    idx : int
        If there are multiple `select` nodes, select the idx-th node. If only one, ignore this parameter.
    idx2 : int
        If there are multiple `cmap` and `clim` for a node, select the idx2-th cmap and clim. If only one, ignore this parameters.
        This parameter is only used when select is 'surface' and 'logs'
    """
    # fmt: off
    assert len(nodes) > 0, "there is no node, len(nodes) == 0"
    if select == 'auto':
        if any([isinstance(node, AxisAlignedImage) and len(node.overlaid_images) > 1 for node in nodes]):
            select = 'mask'
        elif any([isinstance(node, SurfaceNode) for node in nodes]):
            select = 'surface'
        elif any([isinstance(node, AxisAlignedImage) for node in nodes]):
            select = 'slices'
        elif any([isinstance(node, WellLog) for node in nodes]):
            select = 'logs'
        elif any([isinstance(node, Line) for node in nodes]):
            select = 'line_logs'
        elif any([isinstance(node, Mesh) for node in nodes]):
            select = 'mesh'
        else:
            raise ValueError("No valid nodes")
    elif select == 'fault_skin':
        select = 'mesh'

    assert select in ['last', 'mask', 'surface', 'slices', 'logs', 'line_logs', 'mesh']
    cmap = None
    clim = None
    if select == 'mask' or (select == 'last' and isinstance(nodes[-1], AxisAlignedImage) and len(nodes[-1].overlaid_images) > 1):
        if select != 'last':
            node = [node for node in nodes if isinstance(node, AxisAlignedImage)]
            if len(node) == 0 or len(node[0].overlaid_images) == 1:
                raise ValueError(f"No valid nodes, {len(node)} AxisAlignedImage or no mask")
            if len(node[0].overlaid_images) == 2:
                idx = 0
            elif len(node[0].overlaid_images) <= idx + 1:
                raise ValueError(f"idx error, there are only {len(node[0].overlaid_images)-1} mask, but got idx = {idx}")
        else:
            node = [nodes[-1]]
        cmap = node[0].overlaid_images[idx + 1].cmap
        clim = node[0].overlaid_images[idx + 1].clim
    elif select == 'surface' or (select == 'last' and isinstance(nodes[-1], SurfaceNode)):
        if select != 'last':
            node = [node for node in nodes if isinstance(node, SurfaceNode)]
            if len(node) == 0:
                raise ValueError("No valid nodes, no SurfaceNode")
            if len(node) == 1:
                idx = 0
            if len(node) <= idx:
                raise ValueError(f"idx error, there are only {len(node)} SurfaceNode, but got idx = {idx}")
        else:
            node = [nodes[-1]]
            idx = 0
        if len(node[idx].cmaps) == 1:
            idx2 = 0
        if len(node[idx].cmaps) <= idx2:
            raise ValueError(f"idx2 error, there are only {len(node[idx].cmaps)} cmaps for the SurfaceNode, but got idx = {idx2}")
        cmap = node[idx].cmaps[idx2]
        clim = node[idx].clims[idx2]
    elif select == 'slices' or (select == 'last' and isinstance(nodes[-1], AxisAlignedImage) and len(nodes[-1].overlaid_images) == 1):
        if select != 'last':
            node = [node for node in nodes if isinstance(node, AxisAlignedImage)]
            if len(node) == 0:
                raise ValueError("No valid nodes, no AxisAlignedImage")
        else:
            node = [nodes[-1]]
        cmap = node[0].overlaid_images[0].cmap
        clim = node[0].overlaid_images[0].clim
    elif select == 'logs' or (select == 'last' and isinstance(nodes[-1], WellLog)):
        if select != 'last':
            node = [node for node in nodes if isinstance(node, WellLog)]
            if len(node) == 0:
                raise ValueError("No valid nodes, no WellLog")
            if len(node) == 1:
                idx = 0
            if len(node) <= idx:
                raise ValueError(f"idx error, there are only {len(node)} WellLog, but got idx = {idx}")
        else:
            node = [nodes[-1]]
        if len(node[idx].cmap) == 1:
            idx2 = 0
        if len(node[idx].cmap) <= idx2:
            raise ValueError(f"idx2 error, there are only {len(node[idx].cmap)} cmaps for the SurfaceNode, but got idx = {idx2}")
        cmap = node[idx].cmap[idx2]
        clim = node[idx].clim[idx2]
    else:
        if select != 'last':
            raise ValueError(f"select: {select} not support now")
        else:
            raise ValueError(f"last node is {type(nodes[-1])}, which is not support now")
    # fmt: on

    cbar = Colorbar(cmap=cmap, clim=clim, label_str=label_str, **kwargs)
    return [cbar]


def set_surface_color_by_slices_nodes(nodes, volumes):
    if not isinstance(volumes, (List, Tuple)):
        volumes = [volumes]
    alignImage = [node for node in nodes if isinstance(node, AxisAlignedImage)]
    surfNode = [node for node in nodes if isinstance(node, SurfaceNode)]
    if len(surfNode) == 0:
        raise ValueError("The `nodes` don't contain `SurfaceNode`")
    if len(alignImage) == 0:
        raise ValueError("The `nodes` don't contain `AxisAlignedImage`, that means no slice and mask") # yapf: disable
    alignImage = alignImage[0]
    if len(alignImage.overlaid_images) != len(volumes):
        raise ValueError(f"A slice contains {len(alignImage.overlaid_images)} image (base + masks), but got {len(volumes)} volumes") # yapf: disable

    for node in surfNode:
        node.update_colors_by_slice_node([surfNode], volumes)

    return nodes


def create_surfaces(surfs: List[np.ndarray],
                    volume: np.ndarray = None,
                    value_type: str = 'depth',
                    clim: List = None,
                    cmap: str = 'jet',
                    shape: Union[Tuple, List] = None,
                    interp: bool = False,
                    quad: bool = False,
                    step1: int = 1,
                    step2: int = 1,
                    shading: str = 'smooth',
                    dyn_light: bool = True,
                    **kwargs) -> List:
    """
    create a surfaces node

    Parameters
    ----------
    surfs : List or array-like
        the surface position, which can be an array (one surface) or 
        List (multi-surfaces). Each surf can be a (n1, n2)
        array or (N, 3) array, such as
        >>> surf.shape = (n1, n2) # surf[i, j] means z pos at x=i, y=j
        >>> surf.shape = (N, 3) # surf[i, :] means i-th point position
    volume : array-like
        3D array, values when surf_color is 'amp'
    value_type : List of str or ArrayLike
        'depth' for showing z, 'amp' for displaying amplitude of volume, 
        or an array-like for values
    clim : List
        [vmin, vmax] of surface volumes
    cmap : str or Colormap
        cmap for surface
    shape : List or Tuple
        If surf's shape is like (N, 3), shape must be specified,
        if surf's shape is like (n1, n2), shape will be ignored
    interp : bool
        interpolate the surface or not if the surf is not dense
    step1 : int
        mesh interval in x direction
    step2 : int
        mesh interval in y direction
    shading : str
        could be one of ['smooth', 'flat', None], if None, no shading filter
    dyn_light : bool
        dynamic light or not, valid when shading is not None
    """
    utils.check_mmap(volume)

    # add surface
    if not isinstance(surfs, List):
        surfs = [surfs]

    if any([sf.ndim > 2 for sf in surfs]):
        warnings.warn(
            "Passing surfs with ndim > 2, i.e. combining the color matrix "
            "or value directly with the surf, is deprecated. Deprecated "
            "since 0.1.0; scheduled for removal in "
            f"{utils.DEPRECATION_REMOVAL_VERSION}. Put the value or color "
            "matrix in value_type instead. See "
            "`examples/3Dvispy/12-surf-overlay.py` for guidance.",
            DeprecationWarning,
            stacklevel=2,
        )
        surfs = surfs[0] if len(surfs) == 1 else surfs
        return _create_surfaces_old(surfs, volume, value_type, clim, cmap, 1, shape, interp, step1=step1, step2=step2, **kwargs) # yapf: disable

    if isinstance(value_type, str):
        value_type = [value_type] * len(surfs)
    if not isinstance(value_type, List):
        value_type = [value_type]
    if len(surfs) == 1 and len(value_type) > 1:
        value_type = [value_type]
    assert len(value_type) == len(surfs)

    if not isinstance(clim, List):
        clim = [clim] * len(surfs)
    if isinstance(clim, List) and not isinstance(clim[0], List):
        clim = [clim] * len(surfs)
    if len(surfs) == 1 and len(clim) > 1:
        clim = [clim]
    if not isinstance(cmap, List):
        cmap = [cmap] * len(surfs)
    if len(surfs) == 1 and len(cmap) > 1:
        cmap = [cmap]

    nodes = []
    for i in range(len(surfs)):
        node = SurfaceNode(surfs[i],
                           volume,
                           value_type[i],
                           clim[i],
                           cmap[i],
                           shape,
                           step1,
                           step2,
                           interp=interp,
                           quad=quad,
                           shading=shading,
                           dyn_light=dyn_light,
                           **kwargs)
        nodes.append(node)

    return nodes


def _create_surfaces_old(surfs: List[np.ndarray],
                         volume: np.ndarray = None,
                         value_type: str = 'depth',
                         clim: List = None,
                         cmap: str = 'jet',
                         alpha: float = 1,
                         shape: Union[Tuple, List] = None,
                         interp: bool = False,
                         return_cbar: bool = False,
                         step1=1,
                         step2=1,
                         **kwargs) -> List:
    """
    create a surfaces node

    Parameters
    ----------
    surfs : List or array-like
        the surface position, which can be an array (one surface) or 
        List (multi-surfaces). Each surf can be a (n1, n2)/(n1, n2, 2) 
        array or (N, 3)/(N, 4) array, such as
        >>> surf.shape = (n1, n2) # surf[i, j] means z pos at x=i, y=j
        # surf[i, j, 0] means z pos at x=i, y=j
        # surf[i, j, 1] means value for plotting at pos (i, j, surf[i, j])
        >>> surf.shape = (n1, n2, 2)
        # surf[i, j, 1:] means rgb or rgba color at pos (i, j, surf[i, j])
        >>> surf.shape = (n1, n2, 4) or (n1, n2, 5)
        >>> surf.shape = (N, 3) # surf[i, :] means i-th point position
        # surf[i, :3] means i-th point position
        # surf[i, 3] means i-th point's value for plotting
        >>> surf.shape = (N, 4)
        # surf[i, 3:] means i-th point color in rgb or rgba format
        >>> surf.shape = (N, 6) or (N, 7)
    volume : array-like
        3D array, values when surf_color is 'amp'
    value_type : str
        'depth' or 'amp', show z or amplitude, amplitude can be values in volume or
        values or colors
    clim : List
        [vmin, vmax] of surface volumes
    cmap : str or Colormap
        cmap for surface
    alpha : float
        opactity of the surfaces
    shape : List or Tuple
        If surf's shape is like (N, 3) or (N, 4), shape must be specified,
        if surf's shape is like (n1, n2) or (n1, n2, 2), shape will be ignored
    cbar_params : Dict
        params to create a colorbar
    
    kwargs : Dict
        parameters for vispy.scene.visuals.Mesh
    """
    utils.check_mmap(volume)
    utils.check_mmap(surfs)
    line_first = cigvis.is_line_first()
    method = kwargs.pop('method', 'cubic')
    fill = kwargs.pop('fill', -1)
    anti_rot = kwargs.pop('anti_rot', True)

    # add surface
    if not isinstance(surfs, List):
        surfs = [surfs]

    surfaces = []
    values = []
    colors = []
    for surf in surfs:
        if surf.ndim == 3:
            s, v, c = surfaceutils.preproc_surf_array3(surf, value_type)
        elif surf.ndim == 2:
            if surf.shape[1] > 7:
                s, v, c = surfaceutils.preproc_surf_array2(
                    surf, volume, value_type)
            else:
                assert volume is not None or shape is not None
                if shape is None:
                    shape = volume.shape[:2] if line_first else volume.shape[1:]
                s, v, c = surfaceutils.preproc_surf_pos(
                    surf, shape, volume, value_type, interp, method, fill)
        else:
            raise RuntimeError('Invalid shape')
        surfaces.append(s)
        values.append(v)
        colors.append(c)

    if value_type == 'depth':
        values = surfaces

    if clim is None and value_type == 'amp':
        vmin = min([utils.nmin(s) for s in values])
        vmax = max([utils.nmax(s) for s in values])
        clim = [vmin, vmax]
    elif clim is None and value_type == 'depth':
        vmin = min([s[s >= 0].min() for s in values])
        vmax = max([s[s >= 0].max() for s in values])
        clim = [vmin, vmax]

    cmap = colormap.cmap_to_vispy(cmap)
    if alpha < 1:
        cmap = colormap.set_alpha(cmap, alpha)

    mesh_nodes = []
    for s, v, c in zip(surfaces, values, colors):
        mask = np.logical_or(s < 0, np.isnan(s))
        vertices, faces = surface2mesh(
            s,
            mask,
            anti_rot=anti_rot,
            step1=step1,
            step2=step2,
        )
        mask = mask[::step1, ::step2]
        if v is not None:
            v = v[::step1, ::step2]
            v = v[~mask].flatten()
        if c is not None:
            channel = c.shape[-1]
            c = c[::step1, ::step2, ...]
            c = c[~mask].flatten().reshape(-1, channel)

        if kwargs.get('color', None) is not None:
            v = None
            c = None
        if c is not None:
            v = None

        mesh = Mesh(vertices=vertices,
                    faces=faces,
                    vertex_values=v,
                    vertex_colors=c,
                    shading='smooth',
                    **kwargs)

        if v is not None and c is None and kwargs.get('color', None) is None:
            mesh.cmap = cmap
            mesh.clim = clim

        mesh_nodes.append(mesh)

    if return_cbar:
        cbar = create_colorbar(cmap, clim)
        return mesh_nodes, cbar
    return mesh_nodes


def create_bodies(volume: np.ndarray,
                 level: float,
                 margin: float = None,
                 color: str = 'yellow',
                 filter_sigma: Union[float, List] = None,
                 shading: str = 'smooth',
                 dyn_light: bool = True,
                 **kwargs) -> List:
    """
    using marching_cubes to find meshs (its vertices and faces), 
    and the then use vispy.scene.visuals.Mesh to show them

    Parameters
    ----------
    volume : array-like
        3D array
    level : float
        mesh value
    color : str
        color for mesh
    margin : float
        if is not None, set a margin to the volume
    filter_sigma : float
        if is not None, filter the volume by gaussian filter
    shading : str
        could be one of ['smooth', 'flat', None], if None, no shading filter
    dyn_light : bool
        dynamic light or not, valid when shading is not None
    
    kwargs : Dict
        parameters for vispy.scene.visuals.Mesh
    """
    utils.check_mmap(volume)
    if (filter_sigma is not None) or (margin is not None):
        if isinstance(volume, np.memmap):
            assert volume.mode != 'r', \
                "margin will modify the volume, set `mode='c'` " + \
                "instead of `mode='r'` in np.memmap"

    if filter_sigma is not None:
        volume = gaussian_filter(volume, filter_sigma)

    if margin is not None:
        volume[0, :, :] = margin
        volume[:, 0, :] = margin
        volume[:, :, 0] = margin
        volume[volume.shape[0] - 1, :, :] = margin
        volume[:, volume.shape[1] - 1, :] = margin
        volume[:, :, volume.shape[2] - 1] = margin

    # marching_cubes in skimage is more faster
    # F3 demo, salt body, skimage: 3.04s, vispy: 21.44s
    verts, faces, normals, values = marching_cubes(volume, level)
    _shading = None if (dyn_light and shading is not None) else shading
    body = Mesh(verts, faces, color=color, shading=_shading, **kwargs)
    if dyn_light and shading is not None:
        body.attach(HeadlightShadingFilter(shading=shading))
    body.unfreeze()
    body.dyn_light = dyn_light
    body.freeze()

    # HACK, NOTE: use Isosurface or convert to Mesh?
    # body = Isosurface(volume, level=level, color=color, shading='smooth')
    # # must call _prepare_draw before attaching ShadingFilter
    # # see: https://github.com/vispy/vispy/issues/2254#issuecomment-967276060
    # body._prepare_draw(body)

    return [body]


@utils.deprecated(
    "The function was renamed for spelling consistency.",
    "`cigvis.create_bodies`",
)
def create_bodys(*args, **kwargs) -> List:
    """Deprecated alias for :func:`create_bodies`."""
    return create_bodies(*args, **kwargs)


def create_line_logs(logs: Union[List, np.ndarray],
                     value_type: str = 'depth',
                     cmap: str = 'jet',
                     clim: List = None,
                     width: float = 6.0,
                     return_cbar: bool = False,
                     cbar_kw: Dict = None,
                     **kwargs):
    """
    create Line nodes to plot logs data

    Parameters
    ----------
    logs : List or array-like
        List (multi-logs) or np.ndarray (one log). For a log,
        its shape is like (N, 3) or (N, 4) or (N, 6) or (N, 7),
        the first 3 columns are (x, y, z) coordinates. If 3 columns,
        use the third column (z) as the color value (mapped by `cmap`), 
        if 4 columns, the 4-th column is the color value (mapped by `cmap`)
        when value_type is not 'depth', if 6 or 7 columns, colors are 
        RGB or RGBA format when value_type is not 'depth'.
    value_type : str
        'depth' or 'amp', if 'depth', force the colors are mapped by 
        'depth' (z or 3th column).
    cmap : str
        colormap
    clim : List
        [vmin, vmax] for showing
    width : float
        Line width
    return_cbar : bool
        return a colorbar
    
    kwargs : Dict
        parameters for vispy.scene.visuals.Line
    """
    warnings.warn(
        "create_line_logs is discouraged but not scheduled for removal. "
        "Prefer `cigvis.create_well_logs` for new code.",
        UserWarning,
        stacklevel=2,
    )
    if isinstance(logs, np.ndarray):
        assert logs.shape[1] >= 3
        logs = [logs]

    pos = []
    values = []
    for log in logs:
        assert log.ndim == 2 and log.shape[1] >= 3
        pos.append(log[:, :3])
        if value_type == 'depth':
            values.append(log[:, 2])
        else:
            if log.shape[1] == 3:
                values.append(log[:, 2])
            elif log.shape[1] == 4:
                values.append(log[:, 3])
            elif log.shape[1] == 6 or log.shape[1] == 7:
                values.append(log[:, 3:])
            else:
                raise RuntimeError("Invalid shape")

    if clim is None:
        clim = [
            min([utils.nmin(v) for v in values]),
            max([utils.nmax(v) for v in values])
        ]

    log_nodes = []
    for p, v in zip(pos, values):
        if v.ndim == 1:
            v = colormap.get_colors_from_cmap(cmap, clim, v)
        log_nodes.append(Line(p, width=width, color=v, **kwargs))

    if return_cbar:
        cbar = create_colorbar(cmap, clim, **(cbar_kw or {}))
        return log_nodes, cbar
    return log_nodes


@utils.deprecated(
    "The function was renamed to snake_case.",
    "`cigvis.create_line_logs`",
)
def create_Line_logs(*args, **kwargs):
    """Deprecated alias for :func:`create_line_logs`."""
    return create_line_logs(*args, **kwargs)


def create_well_logs(points: np.ndarray,
                     values: np.ndarray = None,
                     cmap: Union[str, List] = 'jet',
                     cyclinder: bool = True,
                     radius_tube: Union[float, List] = 1.5,
                     radius_line: List = [2.2, 5],
                     null_value: float = None,
                     clim: List = None,
                     index: List = None,
                     tube_points: int = 16,
                     mode: str = 'triangles',
                     shading: str = 'smooth',
                     dyn_light: bool = True,
                     **kwargs):
    """
    create a well log node

    Parameters
    -----------
    points : array-like
        points positions, shape as (N, 3)
    values : array-like
        log curves, shape as (N, m), m curves
    cmap : List or str
        colormaps for each curves
    cyclinder : bool
        a cyclinder with a same radius or not 
    radius_tube : float or List
        if cyclinder, it's a float, otherwise a List: [min_radius, max_radius]
    radius_line : List
        the log curves face radius
    null_value : float
        null value of log curves
    clim : List
        [[vmin1, vmax1], [vmin2, vmax2], ...] for log curves
    index : List
        point index of each log curve attached to
    tube_points : int
        the number of points to represent a circle
    mode : str
        use 'triangles'
    shading : str
        could be one of ['smooth', 'flat', None], if None, no shading filter
    dyn_light : bool
        dynamic light or not, valid when shading is not None

    Returns
    --------
    node : List
        List of a `WellLog`
    """
    assert points.ndim == 2 and points.shape[1] == 3
    if values is None:
        values = points[:, 2]
    if values.ndim == 1:
        values = values[:, np.newaxis]

    assert len(points) == len(values)
    nlogs = values.shape[1]
    if not isinstance(cmap, List):
        cmap = [cmap] * nlogs
    assert len(cmap) == nlogs

    if null_value is not None:
        values[values == null_value] = np.nan

    if clim is None:
        clim = [[utils.nmin(values[:, i]),
                 utils.nmax(values[:, i])] for i in range(nlogs)]

    mintube = radius_tube
    if not cyclinder:
        if not isinstance(radius_tube, List):
            print('radius_tube is not a List, set as default: [1, 2]')
            radius_tube = [1, 2]
        mintube = max(radius_tube)

    if values.shape[1] > 1:
        assert min(radius_line) > mintube

    colors = np.zeros((nlogs, len(points), 4), dtype=float)
    radius = []

    def _cal_radius(v, r):
        return r[0] + (v - utils.nmin(v)) / (utils.nmax(v) - utils.nmin(v)) * (r[1] - r[0]) # yapf: disable

    # tube radius and colors
    if cyclinder:
        radius.append(radius_tube)
    else:
        r = _cal_radius(values[:, 0], radius_tube)
        r[np.isnan(r)] = radius_tube[0]
        radius.append(r)

    values[np.isnan(values[:, 0]), 0] = null_value
    colors[0] = colormap.get_colors_from_cmap(cmap[0], clim[0], values[:, 0])

    # line radius and colors
    for i in range(1, nlogs):
        colors[i] = colormap.get_colors_from_cmap(cmap[i], clim[i], values[:, i]) # yapf: disable
        r = _cal_radius(values[:, i], radius_line)
        radius.append(r)

    node = WellLog(points,
                   radius,
                   colors,
                   index,
                   tube_points,
                   mode,
                   shading=shading,
                   dyn_light=dyn_light)
    node.cmap = cmap
    node.clim = clim

    return [node]


def create_points(points: np.ndarray,
                  r: float = 2,
                  color: str = 'green',
                  cmap='jet',
                  clim=None,
                  shading='flat',
                  dyn_light=True,
                  **kwargs):
    """
    create a node to show points using Mesh instead of Marker

    Parameters
    ----------
    points : array-like
        points, shape is like (N, 3).
    r : float
        the radius of a point, to control the size of a point
    color : str
        color to fill
    cmap : str
        colormap to map when set `vertex_values`
    clim : List
        clim if use cmap
    shading : str
        could be one of ['smooth', 'flat', None], if None, no shading filter
    dyn_light : bool
        dynamic light or not, valid when shading is not None

    kwargs : Dict
        parameters for Mesh 
    """
    points = np.array(points)
    assert points.ndim == 2 and points.shape[1] >= 3
    vertices, faces = cigvis.meshs.cube_points(points[:, :3], r)

    if color is not None:
        kwargs['vertex_values'] = None
        kwargs['vertex_colors'] = None
    else:
        vertex_values = kwargs.get('vertex_values', None)
        if vertex_values is not None:
            assert len(vertex_values) == len(points)
            vertex_values = np.array(vertex_values)
            if clim is None:
                clim = [vertex_values.min(), vertex_values.max()]
            kwargs['vertex_values'] = np.repeat(vertex_values, 8)
        vertex_colors = kwargs.get('vertex_colors', None)
        if vertex_colors is not None:
            assert len(vertex_colors) == len(points)
            kwargs['vertex_colors'] = np.repeat(vertex_colors, 8, axis=0)

    _shading = None if (dyn_light and shading is not None) else shading
    point_mesh = Mesh(vertices=vertices,
                      faces=faces,
                      color=color,
                      shading=_shading,
                      **kwargs)
    if dyn_light and shading is not None:
        point_mesh.attach(HeadlightShadingFilter(shading=shading))
    point_mesh.unfreeze()
    point_mesh.dyn_light = dyn_light
    point_mesh.freeze()

    if color is None and kwargs.get('vertex_values', None) is not None:
        point_mesh.cmap = cmap
        point_mesh.clim = clim

    return [point_mesh]


def create_fault_skin(skin_dir,
                      suffix='*',
                      endian='>',
                      values_type='likelihood',
                      cmap='jet',
                      clim=None,
                      shading='smooth',
                      dyn_light=True,
                      **kwargs):
    """"""
    if os.path.isfile(skin_dir):
        vertices, faces, values = cigvis.io.load_one_skin(
            skin_dir, endian, values_type)
    elif os.path.isdir(skin_dir):
        vertices, faces, values = cigvis.io.load_skins(skin_dir, suffix,
                                                       endian, values_type)
    if kwargs.get('color', None) is not None:
        values = None
    if clim is None and values is not None:
        clim = [values.min(), values.max()]

    _shading = None if (dyn_light and shading is not None) else shading
    node = Mesh(vertices,
                faces,
                vertex_values=values,
                shading=_shading,
                **kwargs)
    if dyn_light and shading is not None:
        node.attach(HeadlightShadingFilter(shading=shading))
    node.unfreeze()
    node.dyn_light = dyn_light
    node.freeze()
    if values is not None and kwargs.get('vertex_colors',
                                         None) is None and kwargs.get(
                                             'color', None) is None:
        node.cmap = cmap
        node.clim = clim

    return [node]


def create_arbitrary_line(path=None,
                          anchor=None,
                          data=None,
                          volume=None,
                          nodes=None,
                          cmap='gray',
                          clim=None,
                          hstep=1,
                          vstep=1,
                          **kwargs):
    """
    Create an arbitrary line mesh node. 
    You can pass one of `path` or `anchor` to define the arbitrary line path in X-Y pane.
    You also need to pass one of `data` or `volume` to define arbitrary line values, and if `data` is None, will interpolate from `volume`.
    To show the arbitrary line, you need pass `cmap`, `clim` to define the colors. 
    You can also pass `nodes`, we will use the `cmap` and `clim` of AxisAlignedImage in `nodes` to define the colors, and the `cmap`, `clim` will be ignore.

    Parameters
    ----------
    path : array-like
        The path of the arbitrary line, shape is like (N, 2)
    anchor : array-like
        The anchor of the arbitrary line, shape is like (m, 2), this can be view as the turning endpoints of a folded line. 
        We will interpolate the path between the anchor points.
    data : array-like
        The values of the arbitrary line, shape is like (N, nt)
    volume : array-like
        The 3D volume, shape is like (ni, nx, nt), if data is None, will interpolate from volume
    nodes : List
        The nodes to get the `cmap` and `clim` to define the colors
    cmap : str
        The colormap for the arbitrary line
    clim : List
        The clim for the arbitrary line
    hstep : int
        The horizontal step for the vertices of the arbitrary line mesh
    vstep : int
        The vertical step for the vertices of the arbitrary line mesh
    """
    # TODO: when passing `nodes`, can set multiple data (i.e., base image and mask)?
    if nodes is not None:
        node = [n for n in nodes if isinstance(n, AxisAlignedImage)]
        if len(node) == 0:
            warnings.warn(
                "The passed nodes don't contain `AxisAlignedImage`, so the `cmap` and `clim` will be used",
                UserWarning)
        else:
            cmap = node[0].overlaid_images[0].cmap
            clim = node[0].overlaid_images[0].clim
    return [ArbLineNode(path, anchor, data, volume, cmap, clim, hstep, vstep, **kwargs)] # yapf: disable


def create_axis(
    shape,
    mode='box',
    axis_pos=[3, 3, 1],
    north_direction=None,
    tick_nums=7,
    ticks_font_size=18,
    labels_font_size=20,
    intervals=[1, 1, 1],
    starts=[0, 0, 0],
    axis_labels=['Inline', 'Xline', 'Time'],
    north_scale=2,
    **kwargs,
):
    """
    3D axis with ticks and labels.

    Parameters
    ------------
    shape : tuple
        The bound of the 3D world
    mode : str
        The mode of the axis, 'box' or 'axis'
    axis_pos : list or str
        Which axis to show ticks? axis_pos can be set as 'auto' or a List. If is a List,
        for each axis, it can be 0, 1, 2, 3, 
        representing the starting point of the ticks along the axis.
        0: For 'x' axis -> (0, 0, 0), for 'y' axis -> (0, 0, 0), for 'z' axis -> (0, 0, 0)
        1: For 'x' axis -> (0, 0, nz), for 'y' axis -> (0, 0, nz), for 'z' axis -> (0, ny, 0)
        2: For 'x' axis -> (0, ny, 0), for 'y' axis -> (nx, 0, 0), for 'z' axis -> (nx, 0, 0)
        3: For 'x' axis -> (0, ny, nz), for 'y' axis -> (nx, 0, nz), for 'z' axis -> (nx, ny, 0)
    north_direction : list
        The direction of the north, if not None, will create a `NorthPointer`
    tick_nums : int
        The number of ticks on each axis
    ticks_font_size : int
        The font size of the ticks
    labels_font_size : int
        The font size of the labels
    intervals : list
        The sample intervals of the axis
    starts : list
        The first sample of the axis
    samplings : list[np.ndarray]
        The sample points of the axis, default is None
    axis_labels : list
        The labels of the axis
    """
    axis = Axis3D(shape,
                  mode,
                  axis_pos,
                  tick_nums,
                  ticks_font_size,
                  labels_font_size,
                  intervals,
                  starts,
                  axis_labels=axis_labels,
                  **kwargs)
    nodes = [axis]
    if north_direction is not None:
        assert len(north_direction) == 2
        nodes.append(NorthPointer(north_direction, north_scale))

    return nodes


def _dict_option(value, name: str) -> Dict:
    if value is None:
        return {}
    if is_dataclass(value):
        return {
            field.name: getattr(value, field.name)
            for field in fields(value)
            if getattr(value, field.name) is not None
        }
    if not isinstance(value, dict):
        raise TypeError(
            f"plot3D({name}=...) must be a dict or a cigvis Plot3D* "
            "configuration object."
        )
    return dict(value)


def _save_option(value) -> Dict:
    if value is None:
        return {}
    if isinstance(value, (str, os.PathLike)):
        return {'path': value}
    if is_dataclass(value):
        out = _dict_option(value, 'save')
    elif isinstance(value, dict):
        out = dict(value)
    else:
        raise TypeError(
            "plot3D(save=...) must be a path string, dict, or "
            "cigvis.Plot3DSave."
        )
    if 'directory' in out:
        out['dir'] = out.pop('directory')
    if 'savename' in out:
        out['path'] = out.pop('savename')
    if 'savedir' in out:
        out['dir'] = out.pop('savedir')
    return out


def _gui_option(value) -> Dict:
    if isinstance(value, bool):
        return {'enabled': value}
    if value is None:
        return {'enabled': False}
    if is_dataclass(value):
        out = _dict_option(value, 'gui')
        out.setdefault('enabled', True)
        return out
    if isinstance(value, dict):
        out = dict(value)
        out.setdefault('enabled', True)
        return out
    raise TypeError("plot3D(gui=...) must be a bool, dict, or cigvis.Plot3DGui.")


def _warn_plot3d_legacy(keys: List[str]) -> None:
    if not keys:
        return
    joined = ', '.join(keys)
    warnings.warn(
        "plot3D top-level parameter(s) are deprecated: "
        f"{joined}. Use view={{...}}, save={{...}}, cbar={{...}}, "
        "and gui={...} so each option has a clear owner. Deprecated since "
        f"{utils.DEPRECATION_VERSION}; scheduled for removal in "
        f"{utils.DEPRECATION_REMOVAL_VERSION}.",
        FutureWarning,
        stacklevel=3,
    )


def _build_plot3d_options(view, save, cbar, gui, legacy_kwargs):
    view_cfg = _dict_option(view, 'view')
    save_cfg = _save_option(save)
    cbar_cfg = _dict_option(cbar, 'cbar') if not isinstance(cbar, bool) else {'save': cbar}
    gui_cfg = _gui_option(gui)
    legacy_used = []
    ignored = []

    layout_keys = {'grid', 'share', 'xyz_axis'}
    canvas_keys = {
        'size', 'show', 'bgcolor', 'scale_factor', 'center', 'fov', 'azimuth',
        'elevation', 'zoom_factor', 'axis_scales', 'title', 'keys',
    }
    view_keys = layout_keys | canvas_keys | {'cbar_region_ratio'}

    for key in list(legacy_kwargs):
        value = legacy_kwargs.pop(key)
        if key == 'view':
            view_cfg.update(_dict_option(value, 'view'))
        elif key == 'layout':
            view_cfg.update(_dict_option(value, 'layout'))
            legacy_used.append(key)
        elif key == 'canvas':
            view_cfg.update(_dict_option(value, 'canvas'))
            legacy_used.append(key)
        elif key in view_keys:
            view_cfg[key] = value
            legacy_used.append(key)
        elif key == 'savename':
            save_cfg['path'] = value
            legacy_used.append(key)
        elif key == 'savedir':
            save_cfg['dir'] = value
            legacy_used.append(key)
        elif key == 'save_cbar':
            cbar_cfg['save'] = value
            legacy_used.append(key)
        elif key == 'cbar_name':
            cbar_cfg['name'] = value
            legacy_used.append(key)
        elif key == 'gui_theme':
            gui_cfg['theme'] = value
            legacy_used.append(key)
        elif key == 'canvas_kw':
            view_cfg.update(_dict_option(value, 'canvas_kw'))
            legacy_used.append(key)
        elif key == 'save_kw':
            save_cfg.update(_dict_option(value, 'save_kw'))
            legacy_used.append(key)
        elif key == 'cbar_kw':
            cbar_cfg.update(_dict_option(value, 'cbar_kw'))
            legacy_used.append(key)
        elif key == 'dyn_light':
            ignored.append(key)
        else:
            legacy_kwargs[key] = value

    if legacy_kwargs:
        unknown = ', '.join(sorted(legacy_kwargs))
        raise TypeError(
            "Ambiguous plot3D parameter(s): "
            f"{unknown}. Put view/canvas settings in view={{...}}, save "
            "settings in save={...}, and colorbar settings in cbar={...}."
        )

    _warn_plot3d_legacy(legacy_used)
    if ignored:
        warnings.warn(
            "plot3D(dyn_light=...) no longer controls node lighting. Pass "
            "dyn_light to create_surfaces/create_bodies/create_points/"
            "create_fault_skin when creating the nodes. Deprecated since "
            f"{utils.DEPRECATION_VERSION}; scheduled for removal in "
            f"{utils.DEPRECATION_REMOVAL_VERSION}.",
            FutureWarning,
            stacklevel=3,
        )

    allowed_cbar = {field.name for field in fields(Plot3DColorbar)}
    allowed_gui = {field.name for field in fields(Plot3DGui)}
    view_extra = sorted(set(view_cfg) - view_keys)
    if view_extra:
        raise TypeError(
            f"Unknown plot3D view option(s): {', '.join(view_extra)}. "
            "Use cigvis.Plot3DView to see supported options."
        )
    cbar_extra = sorted(set(cbar_cfg) - allowed_cbar)
    if cbar_extra:
        raise TypeError(
            f"Unknown plot3D cbar option(s): {', '.join(cbar_extra)}. "
            "Use cigvis.Plot3DColorbar to see supported options."
        )
    gui_extra = sorted(set(gui_cfg) - allowed_gui)
    if gui_extra:
        raise TypeError(
            f"Unknown plot3D gui option(s): {', '.join(gui_extra)}. "
            "Use cigvis.Plot3DGui to see supported options."
        )

    grid = view_cfg.pop('grid', None)
    share = bool(view_cfg.pop('share', False))
    xyz_axis = bool(view_cfg.pop('xyz_axis', False))

    size = _normalize_render_size(view_cfg.pop('size', (800, 600)),
                                  "view['size']")
    show_canvas = bool(view_cfg.pop('show', True))
    cbar_region_ratio = float(view_cfg.pop('cbar_region_ratio', 0.125))
    save_cbar = bool(cbar_cfg.pop('save', False))
    cbar_name = cbar_cfg.pop('name', 'cbar.png')
    if not save_cbar:
        cbar_name = None

    save_path = save_cfg.pop('path', None)
    save_dir = str(save_cfg.pop('dir', './'))

    gui_enabled = bool(gui_cfg.get('enabled', False))
    gui_theme = gui_cfg.get('theme', 'dark')

    return {
        'grid': grid,
        'share': share,
        'xyz_axis': xyz_axis,
        'size': size,
        'show_canvas': show_canvas,
        'cbar_region_ratio': cbar_region_ratio,
        'cbar_name': cbar_name,
        'cbar_kw': cbar_cfg,
        'canvas_kw': view_cfg,
        'save_path': save_path,
        'save_dir': save_dir,
        'save_kw': save_cfg,
        'gui_enabled': gui_enabled,
        'gui_theme': gui_theme,
    }


def plot3D(nodes: List,
           *args,
           view: Union[Dict, Plot3DView] = None,
           save: Union[str, Dict, Plot3DSave] = None,
           cbar: Union[bool, Dict, Plot3DColorbar] = None,
           gui: Union[bool, Dict, Plot3DGui] = False,
           run_app: bool = True,
           **kwargs):
    """
    Plot 3D vispy nodes.

    New code should group options by ownership:

    >>> cigvis.plot3D(
    ...     nodes,
    ...     view=cigvis.Plot3DView(
    ...         size=(900, 700),
    ...         grid=(1, 2),
    ...         share=True,
    ...         xyz_axis=False,
    ...         bgcolor='white',
    ...     ),
    ...     save=cigvis.Plot3DSave(
    ...         path='example.png',
    ...         size=(3000, 2000),
    ...         output_policy='fit',
    ...         transparent_bg=True,
    ...     ),
    ...     cbar=cigvis.Plot3DColorbar(save=False),
    ...     gui=cigvis.Plot3DGui(enabled=True, theme='dark'),
    ... )

    Plain dicts are also accepted for ``view``/``save``/``cbar``/``gui``.
    Use ``view={'show': False}`` with ``save={...}`` to save offscreen
    without showing a window.

    Legacy top-level parameters such as ``size=``, ``savename=``, ``grid=``,
    ``layout={...}``, and ``canvas={...}`` are still recognized for now, but
    they emit a migration warning. Unknown loose ``**kwargs`` now raise an
    error because they are ambiguous.
    """
    if args:
        raise TypeError(
            "plot3D accepts only `nodes` positionally. Use "
            "view={'grid': ..., 'share': ..., 'xyz_axis': ...} for view "
            "options."
        )

    opts = _build_plot3d_options(view, save, cbar, gui, kwargs)
    grid = opts['grid']
    share = opts['share']
    xyz_axis = opts['xyz_axis']
    size = opts['size']
    show_canvas = opts['show_canvas']
    cbar_region_ratio = opts['cbar_region_ratio']
    cbar_name = opts['cbar_name']
    cbar_kw = opts['cbar_kw']
    canvas_kw = opts['canvas_kw']
    save_path = opts['save_path']
    save_dir = opts['save_dir']
    save_kw = opts['save_kw']

    if grid is None:
        w, h = size
    else:
        h = size[1] / grid[0]
        w = size[0] / grid[1]
    cbar_size = (w * cbar_region_ratio, h)

    # find cbars
    cbar_list = []
    if isinstance(nodes, Dict):
        for k, v in nodes.items():
            cbars = [(i, n) for i, n in enumerate(v)
                     if isinstance(n, Colorbar)]
            if len(cbars) > 1:
                warnings.warn(
                    "only support one colorbar in each subcanvas, so we select the last one"
                )
                out = [nodes[k].pop(i[0])
                       for i in cbars[:-1]]  # remove the other cbars
            if len(cbars) > 0:
                cbars = [cbars[-1][1]]
                cbar_list += cbars
            if xyz_axis:
                nodes[k].append(XYZAxis())
    elif isinstance(nodes[0], List):
        for i, v in enumerate(nodes):
            cbars = [(i, n) for i, n in enumerate(v)
                     if isinstance(n, Colorbar)]
            if len(cbars) > 1:
                warnings.warn(
                    "only support one colorbar in each subcanvas, so we select the last one"
                )
                out = [nodes[i].pop(k[0])
                       for k in cbars[:-1]]  # remove the other cbars
            if len(cbars) > 0:
                cbars = [cbars[-1][1]]
                cbar_list += cbars
            if xyz_axis:
                nodes[i].append(XYZAxis())
    else:
        cbar_list = [(i, n) for i, n in enumerate(nodes)
                     if isinstance(n, Colorbar)]
        if len(cbar_list) > 1:
            warnings.warn(
                "only support one colorbar in each canvas, so we select the last one"
            )
            out = [nodes.pop(k[0])
                   for k in cbar_list[:-1]]  # remove the other cbars
        if len(cbar_list) > 0:
            cbar_list = [cbar_list[-1][1]]
        if xyz_axis:
            nodes.append(XYZAxis())

    # update cbars' size
    for cbar_node in cbar_list:
        cbar_node.update_params(cbar_size=cbar_size,
                                savedir=save_dir,
                                cbar_name=cbar_name,
                                **cbar_kw)

    if opts['gui_enabled']:
        from cigvis.gui.gui3d import gui3d as _gui3d

        gui_canvas_kw = dict(canvas_kw)
        gui_canvas_kw.update({
            'size': size,
            'cbar_region_ratio': cbar_region_ratio,
            'savedir': save_dir,
        })
        win = _gui3d(
            nodes=nodes,
            grid=grid,
            share=share,
            theme=opts['gui_theme'],
            canvas_kwargs=gui_canvas_kw,
            run_app=False,
        )
        if save_path is not None:
            _save_canvas_png(win.canvas.canvas, save_path, save_dir, save_kw)
        if run_app:
            vispy.app.run()
        return win

    canvas_obj = VisCanvas(visual_nodes=nodes,
                           grid=grid,
                           share=share,
                           cbar_region_ratio=cbar_region_ratio,
                           savedir=save_dir,
                           size=size,
                           **canvas_kw)

    if show_canvas:
        canvas_obj.show()

    if save_path is not None:
        _save_canvas_png(canvas_obj, save_path, save_dir, save_kw)

    if run_app and show_canvas:
        vispy.app.run()

    return canvas_obj


def run():
    vispy.app.run()
