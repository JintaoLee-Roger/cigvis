import time
import warnings

from typing import List, Dict, Tuple, Union
import re
import matplotlib.pyplot as plt
import numpy as np
import viser
from PIL import Image, ImageDraw
import imageio.v3 as iio
from packaging import version
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

import cigvis
from cigvis import colormap
from cigvis.visernodes import (
    VolumeSlice,
    PosObserver,
    SurfaceNode,
    MeshNode,
    LogPoints,
    LogLineSegments,
    LogBase,
    Server,
)
from cigvis.meshs import surface2mesh
import cigvis.utils as utils
from cigvis.utils import surfaceutils
from itertools import combinations


def _is_volume_sequence(value) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    if len(value) == 0:
        return False
    return all(hasattr(item, 'ndim') or type(item).__module__ == 'torch' for item in value)


def _normalize_single_value(value, name):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    if isinstance(value, (list, tuple)) and len(value) > 1:
        raise ValueError(f"add_mask only accepts one {name}; call add_mask repeatedly for multiple masks")
    return value


def _normalize_cmap_value(value):
    if isinstance(value, dict):
        return value
    return _normalize_single_value(value, "cmap")


def _normalize_clim_value(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        return value[0]
    return value


def create_slices(volume: np.ndarray,
                  pos: Union[List, Dict] = None,
                  clim: List = None,
                  cmap: str = 'Petrel',
                  nancolor=None,
                  intersection_lines: bool = True,
                  line_color='white',
                  line_width=1,
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
        colormap, it can be str or matplotlib's Colormap
    nancolor : str or color
        color for nan values, default is None (i.e., transparent)
    """
    # set pos
    # ni, nx, nt = volume.shape
    shape, _ = utils.get_shape(volume, cigvis.is_line_first())
    nt = shape[2]
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
        clim = utils.auto_clim(volume)

    nodes = []
    for axis, p in pos.items():
        for i in p:
            nodes.append(
                VolumeSlice(
                    volume,
                    axis,
                    i,
                    cmap,
                    clim,
                    nancolor=nancolor,
                    **kwargs,
                ))

    if intersection_lines:
        observer = PosObserver(color=line_color, width=line_width)
        for node in nodes:
            observer.link_image(node)

    return nodes


def add_mask(nodes: List,
             volume: Union[List, np.ndarray],
             clim: Union[List, Tuple] = None,
             cmap: Union[str, Dict] = None,
             alpha=None,
             excpt=None,
             *,
             clims: Union[List, Tuple] = None,
             cmaps: Union[str, Dict] = None,
             **kwargs) -> List:
    """
    Add Mask/Overlay volumes
    
    Parameters
    -----------
    nodes: List[Node]
        A List that contains `AxisAlignedImage` (may be created by `create_slices`)
    volume : array-like
        3D foreground volume/mask. Add multiple masks by calling add_mask
        repeatedly.
    clim : List
        [vmin, vmax] for foreground slices plotting
    cmap : str, Dict, or Colormap
        colormap for foreground slices. A dict such as ``{'x': 'Reds',
        'z': 'Blues'}`` applies the mask only to those axes.
    alpha : float or List[float]
        if alpha is not None, using `colormap.fast_set_cmap` to set cmap
    excpt : None or str
        it could be one of [None, 'min', 'max', 'ramp']

    Returns
    -------
    slices_nodes : List
        list of slice nodes
    """
    if cmaps is not None:
        if cmap is not None:
            raise ValueError("Specify only one of 'cmap' or deprecated 'cmaps'")
        warnings.warn(
            "'cmaps' is deprecated; use 'cmap' instead.",
            FutureWarning,
            stacklevel=2,
        )
        cmap = cmaps
    if clims is not None:
        if clim is not None:
            raise ValueError("Specify only one of 'clim' or 'clims'")
        clim = clims

    if _is_volume_sequence(volume):
        if len(volume) != 1:
            raise ValueError(
                "add_mask no longer accepts multiple volumes. "
                "Call add_mask repeatedly, for example: "
                "nodes = viserplot.add_mask(nodes, rgt, cmap='stratum'); "
                "nodes = viserplot.add_mask(nodes, fault, cmap='jet')"
            )
        volume = volume[0]

    alpha = _normalize_single_value(alpha, "alpha")
    excpt = _normalize_single_value(excpt, "excpt")
    cmap = _normalize_cmap_value(cmap)
    clim = _normalize_clim_value(clim)
    utils.check_mmap(volume)

    if cmap is None:
        raise ValueError("'cmap' cannot be None")
    if clim is None:
        clim = utils.auto_clim(volume)

    def _prepare_cmap(cmap_value):
        cmap_value = colormap.get_cmap_from_str(cmap_value)
        if alpha is not None:
            cmap_value = colormap.fast_set_cmap(cmap_value, alpha, excpt)
        return cmap_value

    if isinstance(cmap, dict):
        cmap_by_axis = {}
        for axis, cmap_value in cmap.items():
            axis = str(axis).lower()
            if axis not in ('x', 'y', 'z'):
                raise ValueError("cmap dict keys must be one of 'x', 'y', or 'z'")
            cmap_by_axis[axis] = _prepare_cmap(cmap_value)
    else:
        cmap_by_axis = None
        cmap_value = _prepare_cmap(cmap)

    for node in nodes:
        if not isinstance(node, VolumeSlice):
            continue
        if cmap_by_axis is not None:
            if node.axis not in cmap_by_axis:
                continue
            node_cmap = cmap_by_axis[node.axis]
        else:
            node_cmap = cmap_value
        node.add_mask(
            volume,
            node_cmap,
            clim,
        )

    return nodes


def create_surfaces(surfs: List[np.ndarray],
                    volume: np.ndarray = None,
                    value_type: str = 'depth',
                    clim: List = None,
                    cmap: str = 'jet',
                    alpha: float = 1,
                    shape: Union[Tuple, List] = None,
                    interp: bool = False,
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
        vmin = min([utils.nmin(s) for s in values])
        vmax = max([utils.nmax(s) for s in values])
        clim = [vmin, vmax]
    elif clim is None and value_type == 'depth':
        vmin = min([s[s >= 0].min() for s in values])
        vmax = max([s[s >= 0].max() for s in values])
        clim = [vmin, vmax]

    cmap = colormap.get_cmap_from_str(cmap)
    if alpha < 1:
        cmap = colormap.set_alpha(cmap, alpha, False)

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

        mesh_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in ('method', 'fill', 'anti_rot')
        }

        if kwargs.get('color', None) is not None:
            v = None
            c = None
        if c is not None:
            v = None

        mesh = SurfaceNode(vertices=vertices,
                           faces=faces,
                           face_colors=None,
                           vertex_colors=c,
                           vertices_values=v,
                           **mesh_kwargs)

        if v is not None and c is None and kwargs.get('color', None) is None:
            mesh.cmap = cmap
            mesh.clim = clim

        mesh_nodes.append(mesh)

    return mesh_nodes


def create_bodies(volume: np.ndarray,
                  level: float,
                  margin: float = None,
                  color: str = 'yellow',
                  filter_sigma: Union[float, List] = None,
                  **kwargs) -> List:
    """
    Create isosurface body nodes for the viser backend.

    Parameters
    ----------
    volume : array-like
        3D volume used by marching cubes.
    level : float
        Isovalue extracted from ``volume``.
    margin : float
        If not None, set a boundary value before marching cubes.
    color : str or RGB/RGBA
        Uniform body color.
    filter_sigma : float or list
        Optional Gaussian smoothing before marching cubes.
    """
    utils.check_mmap(volume)
    if (filter_sigma is not None) or (margin is not None):
        if isinstance(volume, np.memmap) and volume.mode == 'r':
            raise ValueError(
                "margin/filter_sigma requires writable or copy-on-write data; "
                "open memmap with mode='c' or pass a normal ndarray."
            )
        volume = np.asarray(volume).copy()

    if filter_sigma is not None:
        volume = gaussian_filter(volume, filter_sigma)

    if margin is not None:
        volume[0, :, :] = margin
        volume[:, 0, :] = margin
        volume[:, :, 0] = margin
        volume[-1, :, :] = margin
        volume[:, -1, :] = margin
        volume[:, :, -1] = margin

    vertices, faces, _, _ = marching_cubes(volume, level)
    return [
        MeshNode(
            vertices=vertices,
            faces=faces,
            color=color,
            **kwargs,
        )
    ]


def create_points(points: np.ndarray,
                  r: float = 2,
                  color='green',
                  cmap='jet',
                  clim=None,
                  point_shape='circle',
                  values=None,
                  colors=None,
                  **kwargs) -> List:
    """
    Create sparse point-cloud nodes for horizons, fault points, or picks.

    ``points`` can be ``(N, 3)``, ``(N, 4)`` with the 4th column as scalar
    values, or ``(N, 6)/(N, 7)`` with RGB/RGBA colors. When ``color`` is not
    None, it is used as a uniform color and scalar/color columns are ignored.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] not in (3, 4, 6, 7):
        raise ValueError("points must have shape (N, 3), (N, 4), (N, 6), or (N, 7)")

    if color is None:
        if values is None and colors is None:
            if points.shape[1] == 4:
                values = points[:, 3]
            elif points.shape[1] in (6, 7):
                colors = points[:, 3:]
            else:
                values = points[:, 2]
    else:
        values = None
        colors = None

    return [
        LogPoints(
            points[:, :3],
            values=values,
            colors=colors,
            cmap=cmap,
            clim=clim,
            point_size=r,
            point_shape=point_shape,
            color=color,
            **kwargs,
        )
    ]


def create_well_logs(
    logs: Union[List, np.ndarray],
    logs_type: str = 'point',
    cmap: str = 'jet',
    clim: List = None,
    width: float = 1,
    point_shape: str = 'square',
    **kwargs,
):
    """
    create well logs nodes

    Parameters
    ----------
    logs : List or array-like
        List (multi-logs) or np.ndarray (one log). For a log,
        its shape is like (N, 3) or (N, 4) or (N, 6) or (N, 7),
        the first 3 columns are (x, y, z) coordinates. If 3 columns,
        use the third column (z) as the color value (mapped by `cmap`), 
        if 4 columns, the 4-th column is the color value (mapped by `cmap`),
        if 6 or 7 columns, colors are RGB format.
    logs_type : str
        'point' or 'line', draw points or line segments
    cmap : str
        colormap for logs
    clim : List
        [vmin, vmax] of logs
    width : float
        width of line segments or points
    point_shape : str
        point shape for points, 'square', 'circle' or others, only when logs_type is 'point'
    
    """
    if not isinstance(logs, List):
        logs = [logs]

    nodes = []
    for log in logs:
        assert log.ndim == 2 and log.shape[1] in [3, 4, 6, 7]
        points = log[:, :3]
        values = None
        colors = None
        if log.shape[1] == 3:
            values = log[:, 2]
        elif log.shape[1] == 4:
            values = log[:, 3]
        else:
            colors = log[:, 3:]

        if logs_type == 'line':
            logs = LogLineSegments
        else:
            logs = LogPoints
        nodes.append(
            logs(
                points,
                values,
                colors,
                cmap,
                clim,
                width,
                point_shape=point_shape,
            ))

    return nodes


def plot3D(
    nodes,
    axis_scales=[1, 1, 1],
    fov=30,
    look_at=None,
    wxyz=None,
    position=None,
    server=None,
    run_app=True,
    **kwargs,
):
    if server is None:
        server = Server(label='cigvis-viser', port=8080, verbose=False)
    server.reset()
    server.init_from_nodes(nodes, axis_scales, fov, look_at, wxyz, position)

    if run_app and not cigvis.is_running_in_notebook():
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            server.stop()
            del server
            print("Execution interrupted")


def run():
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Execution interrupted")


def create_server(port=8080, label='cigvis-viser', verbose=False):
    return Server(label=label, port=port, verbose=verbose)


def link_servers(servers):
    """
    Linking Multiple Server Instances to Each Other
    """
    if not all(isinstance(s, Server) for s in servers):
        raise ValueError("Each element must be instance of `Server`.")
    
    for s1, s2 in combinations(servers, 2):
        s1.link(s2)
        s2.link(s1)
