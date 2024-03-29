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

from typing import Callable, List, Tuple, Dict, Union
import warnings
import os
import numpy as np
from cigvis.vispynodes import (
    VisCanvas,
    volume_slices,
    AxisAlignedImage,
    Colorbar,
    WellLog,
    XYZAxis,
)

from vispy.scene.visuals import Mesh, Line
import vispy
from vispy.gloo.util import _screenshot
from scipy.ndimage import gaussian_filter

import cigvis
from cigvis import colormap
from cigvis.utils import surfaceutils
from cigvis.utils import vispyutils
import cigvis.utils as utils

__all__ = [
    "create_slices",
    "add_mask",
    "create_overlay",
    "create_colorbar",
    "create_surfaces",
    "create_bodys",
    "create_Line_logs",
    "create_well_logs",
    "create_points",
    "create_fault_skin",
    "plot3D",
    "run",
]


def create_slices(volume: np.ndarray,
                  pos: Union[List, Dict] = None,
                  clim: List = None,
                  cmap: str = 'Petrel',
                  interpolation: str = 'cubic',
                  return_cbar: bool = False,
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
    return_cbar : bool
        return a colorbar

    kwargs : Dict
        other kwargs for `volume_slices`

    Returns
    -------
    slices_nodes : List
        list of slice nodes
    """
    utils.check_mmap(volume)
    line_first = cigvis.is_line_first()
    if line_first:
        nt = volume.shape[2]
    else:
        nt = volume.shape[0]

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

    if clim is None:
        clim = [volume.min(), volume.max()]
    cmap = colormap.cmap_to_vispy(cmap)

    nodes = volume_slices(volume,
                          pos['x'],
                          pos['y'],
                          pos['z'],
                          cmaps=cmap,
                          clims=clim,
                          interpolation=interpolation)

    if return_cbar:
        cbar_kwargs = vispyutils.get_valid_kwargs('colorbar', **kwargs)
        cbar = create_colorbar(cmap, clim, **cbar_kwargs)
        return nodes, cbar

    return nodes


def add_mask(nodes: List,
             volumes: Union[List[np.ndarray], np.ndarray],
             clims: Union[List, Tuple] = None,
             cmaps: Union[str, List] = None,
             interpolation: str = 'linear',
             method: str = 'auto',
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
        clims = [[v.min(), v.max()] for v in volumes]
    if not isinstance(clims[0], (List, Tuple)):
        clims = [clims]

    if cmaps is None:
        raise ValueError("'cmaps' cannot be 'None'")
    if not isinstance(cmaps, List):
        cmaps = [cmaps] * len(volumes)
    for i in range(len(cmaps)):
        cmaps[i] = colormap.cmap_to_vispy(cmaps[i])

    if isinstance(interpolation, str):
        interpolation = [interpolation] * len(volumes)
    if not isinstance(preproc_funcs, List):
        preproc_funcs = [preproc_funcs] * len(volumes)

    shape = volumes[0].shape
    line_first = cigvis.is_line_first()
    if not line_first:
        shape = shape[::-1]

    for node in nodes:
        if not isinstance(node, AxisAlignedImage):
            continue
        for i in range(len(volumes)):
            node.add_mask(
                volumes[i],
                cmaps[i],
                clims[i],
                interpolation[i],
                method,
                preproc_funcs[i],
            )

    return nodes


@utils.deprecated(
    "The code is based on 'vispy' backbend, and this function will be removed in the feature version.\nExample:\nnodes=cigvis.create_overlay(bg, [fg1, fg2], fg_cmap=['jet', 'gray']\n==> change to ==>\nnodes=cigvis.create_slices(bg)\nnodes=cigvis.add_mask(nodes, [fg1, fg2], cmaps=['jet', 'gray'])\n",
    "`cigvis.add_mask`")
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
    create a overlied slice node

    Parameters
    ----------
    bg_volume : array-like
        3D array, background volume
    fg_volume : array-like or List
        3D array(s), foreground volume(s)
    pos : List or Dict
        init position of the slices, can be a List or Dict, such as:
        ```
        pos = [0, 0, 200] # x: 0, y: 0, z: 200
        pos = [[0, 200], [9], []] # x: 0 and 200, y: 9, z: None
        pos = {'x': [0, 200], 'y': [1], z: []}
        ```
    bg_clim : List
        [vmin, vmax] for background slices plotting
    fg_clim : List
        [vmin, vmax] for foreground slices plotting
    bg_cmap : str or Colormap
        colormap for background slices, it can be str or matplotlib's Colormap or vispy's Colormap
    fg_cmap : str or Colormap
        colormap for foreground slices, it can be str or matplotlib's Colormap or vispy's Colormap
    bg_interpolation : str
        interpolation method for background slices. 
    fg_interpolation : str
        interpolation method. If the values of the slices is discrete, we recommand 
        set as 'nearest'
    return_cbar : bool
        return a colorbar
    cbar_type : str
        'fg' for foreground colorbar, 'bg' for background colorbar
    kwargs : Dict
        other kwargs for `volume_slices`

    Returns
    -------
    slices_nodes : List
        list of slice nodes
    """
    # check
    utils.check_mmap(bg_volume)
    if not isinstance(fg_volume, List):
        fg_volume = [fg_volume]
    for volume in fg_volume:
        assert bg_volume.shape == volume.shape
        utils.check_mmap(volume)

    line_first = cigvis.is_line_first()
    if line_first:
        nt = bg_volume.shape[2]
    else:
        nt = bg_volume.shape[0]

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

    if bg_clim is None:
        bg_clim = [bg_volume.min(), bg_volume.max()]
    if fg_clim is None:
        fg_clim = [[v.min(), v.max()] for v in fg_volume]
    if not isinstance(fg_clim[0], (List, Tuple)):
        fg_clim = [fg_clim]

    bg_cmap = colormap.cmap_to_vispy(bg_cmap)

    if fg_cmap is None:
        raise ValueError("'fg_cmap' cannot be 'None'")
    if not isinstance(fg_cmap, List):
        fg_cmap = [fg_cmap] * len(fg_volume)
    for i in range(len(fg_cmap)):
        fg_cmap[i] = colormap.cmap_to_vispy(fg_cmap[i])

    if isinstance(fg_interpolation, str):
        fg_interpolation = [fg_interpolation] * len(fg_volume)

    nodes = volume_slices([bg_volume, *fg_volume],
                          pos['x'],
                          pos['y'],
                          pos['z'],
                          cmaps=[bg_cmap, *fg_cmap],
                          clims=[bg_clim, *fg_clim],
                          interpolation=[bg_interpolation, *fg_interpolation])

    if return_cbar:
        if cbar_type == 'fg':
            cmap = fg_cmap
            clim = fg_clim
        else:
            cmap = bg_cmap
            clim = bg_clim

        kwargs = vispyutils.get_valid_kwargs('colorbar', **kwargs)

        discrete = kwargs.get('discrete', False)
        disc_ticks = kwargs.get('disc_ticks', None)

        if discrete and disc_ticks is None:
            v = np.unique(fg_volume)
            kwargs['disc_ticks'] = [v]

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

    kwargs = vispyutils.get_valid_kwargs('colorbar', **kwargs)
    cbar = Colorbar(cmap=cmap,
                    clim=clim,
                    discrete=discrete,
                    disc_ticks=disc_ticks,
                    label_str=label_str,
                    **kwargs)

    return cbar


def create_surfaces(surfs: List[np.ndarray],
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
    method = kwargs.get('method', 'cubic')
    fill = kwargs.get('fill', -1)
    anti_rot = kwargs.get('anti_rot', True)

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
        vmin = min([np.nanmin(s) for s in values])
        vmax = max([np.nanmax(s) for s in values])
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
        vertices, faces = surfaceutils.get_vertices_and_faces(
            s, mask, anti_rot=anti_rot, step1=step1, step2=step2)
        mask = mask[::step1, ::step2]
        if v is not None:
            v = v[::step1, ::step2]
            v = v[~mask].flatten()
        if c is not None:
            channel = c.shape[-1]
            c = c[::step1, ::step2, ...]
            c = c[~mask].flatten().reshape(-1, channel)

        mesh_kwargs = vispyutils.get_valid_kwargs('mesh', **kwargs)

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
                    **mesh_kwargs)

        if v is not None and c is None and kwargs.get('color', None) is None:
            mesh.cmap = cmap
            mesh.clim = clim

        mesh_nodes.append(mesh)

    if return_cbar:
        kwargs = vispyutils.get_valid_kwargs('colorbar', **kwargs)
        cbar = create_colorbar(cmap, clim, **kwargs)
        return mesh_nodes, cbar
    return mesh_nodes


def create_bodys(volume: np.ndarray,
                 level: float,
                 margin: float = None,
                 color: str = 'yellow',
                 filter_sigma: Union[float, List] = None,
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

    from skimage.measure import marching_cubes
    # marching_cubes in skimage is more faster
    # F3 demo, salt body, skimage: 3.04s, vispy: 21.44s
    verts, faces, normals, values = marching_cubes(volume, level)
    kwargs = vispyutils.get_valid_kwargs('mesh', **kwargs)
    body = Mesh(verts, faces, color=color, shading='smooth', **kwargs)

    # HACK, NOTE: use Isosurface or convert to Mesh?
    # body = Isosurface(volume, level=level, color=color, shading='smooth')
    # # must call _prepare_draw before attaching ShadingFilter
    # # see: https://github.com/vispy/vispy/issues/2254#issuecomment-967276060
    # body._prepare_draw(body)

    return [body]


def create_Line_logs(logs: Union[List, np.ndarray],
                     value_type: str = 'depth',
                     cmap: str = 'jet',
                     clim: List = None,
                     width: float = 6.0,
                     return_cbar: bool = False,
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
    warnings.warn("We recommand use 'create_well_logs' instead", UserWarning)
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
            min([np.nanmin(v) for v in values]),
            max([np.nanmax(v) for v in values])
        ]

    log_nodes = []
    line_kwargs = vispyutils.get_valid_kwargs('line', **kwargs)
    for p, v in zip(pos, values):
        if v.ndim == 1:
            v = colormap.get_colors_from_cmap(cmap, clim, v)
        log_nodes.append(Line(p, width=width, color=v, **line_kwargs))

    if return_cbar:
        kwargs = vispyutils.get_valid_kwargs('colorbar', **kwargs)
        cbar = create_colorbar(cmap, clim, **kwargs)
        return log_nodes, cbar
    return log_nodes


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
                     mode: str = 'triangles'):
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
        clim = [[np.nanmin(values[:, i]),
                 np.nanmax(values[:, i])] for i in range(nlogs)]

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
        return r[0] + (v - np.nanmin(v)) / (np.nanmax(v) -
                                            np.nanmin(v)) * (r[1] - r[0])

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
        colors[i] = colormap.get_colors_from_cmap(cmap[i], clim[i], values[:,
                                                                           i])
        r = _cal_radius(values[:, i], radius_line)
        radius.append(r)

    node = WellLog(points, radius, colors, index, tube_points, mode)

    return [node]


def create_points(points: np.ndarray,
                  r: float = 2,
                  color: str = 'green',
                  cmap='jet',
                  clim=None,
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

    mesh_kwargs = vispyutils.get_valid_kwargs('mesh', **kwargs)
    point_mesh = Mesh(vertices=vertices,
                      faces=faces,
                      color=color,
                      shading='flat',
                      **mesh_kwargs)

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

    node = Mesh(vertices,
                faces,
                vertex_values=values,
                shading='smooth',
                **kwargs)
    if values is not None and kwargs.get('vertex_colors',
                                         None) is None and kwargs.get(
                                             'color', None) is None:
        node.cmap = cmap
        node.clim = clim

    return [node]


def plot3D(nodes: List,
           grid: Tuple = None,
           share: bool = False,
           xyz_axis: bool = True,
           cbar_region_ratio: float = 0.125,
           savename: str = None,
           savedir: str = './',
           save_cbar: bool = False,
           cbar_name: str = 'cbar.png',
           size: Tuple = (800, 600),
           run_app: bool = True,
           **kwargs):
    """
    plot nodes in a 3D canvas

    Parameters
    -----------
    nodes : List of VisualNodes
        VisualNodes
    grid : Tuple
        grid of the canvas
    share : bool
        link all cameras when grid is not None
    xyz_axis : bool
        add a xyz_axis to each canvas
    cbar_region_ratio : float
        colorbar region ration, i.e., (width*cbar_region_ratio, hight)
    savename : str
        if is not None, will save the figure when rendering
    save_cbar : bool
        save colorbar individual if there is any `Colorbar` in nodes
    cbar_name : str
        the save colorbar image name
    size : Tuple
        canvas size
    kwargs : Dict
        other parameters pass to `Colorbar` and `VisCanvas`
    
    Examples
    ----------
    >>> node1, node2 = [mesh1, mesh2, image1], [mesh3, image2, image3]
    >>> plot3D(node1) # one subcanvas in the canvas
    >>> plot3D([node1, node2], grid=(1, 2)) # two subcanvas in the canvas
    """
    if grid is None:
        w, h = size
    else:
        h = size[1] / grid[0]
        w = size[0] / grid[1]
    cbar_size = (w * cbar_region_ratio, h)
    if not save_cbar:
        cbar_name = None

    # find cbars
    cbar_kwargs = vispyutils.get_valid_kwargs('colorbar', **kwargs)
    cbar_list = []
    if isinstance(nodes, Dict):
        for k, v in nodes.items():
            cbar_list += [n for n in v if isinstance(n, Colorbar)]
            if xyz_axis:
                nodes[k].append(XYZAxis())
    elif isinstance(nodes[0], List):
        for i, v in enumerate(nodes):
            cbar_list += [n for n in v if isinstance(n, Colorbar)]
            if xyz_axis:
                nodes[i].append(XYZAxis())
    else:
        cbar_list = [n for n in nodes if isinstance(n, Colorbar)]
        if xyz_axis:
            nodes.append(XYZAxis())

    # update cbars' size
    for cbar in cbar_list:
        cbar.update_params(cbar_size=cbar_size,
                           savedir=savedir,
                           cbar_name=cbar_name,
                           **cbar_kwargs)

    kwargs = vispyutils.get_valid_kwargs('viscanvas', **kwargs)
    canvas = VisCanvas(visual_nodes=nodes,
                       grid=grid,
                       share=share,
                       cbar_region_ratio=cbar_region_ratio,
                       savedir=savedir,
                       size=size,
                       **kwargs)

    canvas.show()

    if savename is not None:
        screen_shot = _screenshot()
        vispy.io.write_png(savedir + savename, screen_shot)

    if run_app:
        vispy.app.run()


def run():
    vispy.app.run()
