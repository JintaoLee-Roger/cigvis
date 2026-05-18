import time
import warnings

from typing import List, Dict, Tuple, Union
import re
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
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
    GaussianSplatNode,
    Server,
)
from cigvis.meshs import surface2mesh
import cigvis.utils as utils
from cigvis.utils import surfaceutils
from cigvis.utils.slice_provider import SliceProvider
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
    volume : array-like or dict
        3D array, or an axis source dict such as
        ``{'x': iline_source, 'y': xline_source, 'z': time_source}``.
        Each source value may also be a spec such as
        ``{'data': time_source, 'axes': ('z', 'y', 'x')}``.
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
    provider = SliceProvider(
        volume,
        transpose_line_first=False,
        transpose_rgb=False,
    )
    shape = provider.shape
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
        clim = utils.auto_clim(provider.clim_source)

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
                    provider=provider,
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
    volume : array-like or dict
        3D foreground volume/mask, or an axis source dict such as
        ``{'x': iline_source, 'y': xline_source, 'z': time_source}``.
        Each source value may also be a spec such as
        ``{'data': time_source, 'axes': ('z', 'y', 'x')}``.
        Add multiple masks by calling add_mask repeatedly.
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
    provider = SliceProvider(
        volume,
        transpose_line_first=False,
        transpose_rgb=False,
    )

    if cmap is None:
        raise ValueError("'cmap' cannot be None")
    if clim is None:
        clim = utils.auto_clim(provider.clim_source)

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
            provider=provider,
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


_POINT_CLOUD_MODE_PRESETS = {
    'point': {
        'size': 5.0,
        'point_shape': 'circle',
        'point_shading': 'gradient',
        'precision': 'float16',
    },
    'surface': {
        'size': 2.2,
        'point_shape': 'circle',
        'point_shading': 'gradient',
        'precision': 'float16',
    },
    'volume': {
        'size': 2.0,
        'point_shape': 'circle',
        'point_shading': 'gradient',
        'precision': 'float16',
    },
}


def _point_cloud_mode(mode: str) -> str:
    aliases = {
        'point': 'point',
        'points': 'point',
        'pick': 'point',
        'picks': 'point',
        'surface': 'surface',
        'surf': 'surface',
        'splat': 'surface',
        'splats': 'surface',
        'volume': 'volume',
        'voxel': 'volume',
        'voxels': 'volume',
    }
    key = aliases.get(str(mode).lower())
    if key is None:
        raise ValueError("mode must be 'point', 'surface', or 'volume'")
    return key


def _sample_point_cloud_inputs(pos, values, colors, size, max_points, seed):
    if max_points is None or len(pos) <= max_points:
        return pos, values, colors, size

    max_points = int(max_points)
    if max_points <= 0:
        raise ValueError("max_points must be positive")

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pos), max_points, replace=False)
    pos = pos[idx]

    if values is not None:
        values = np.asarray(values)[idx]
    if colors is not None:
        colors = np.asarray(colors)[idx]

    return pos, values, colors, size


def create_splats(pos: np.ndarray,
                  values: np.ndarray = None,
                  cmap: str = 'viridis',
                  clim: List = None,
                  color=None,
                  size=None,
                  mode: str = 'surface',
                  point_shape: str = None,
                  point_shading: str = None,
                  precision: str = None,
                  max_points: int = None,
                  seed: int = 0,
                  **kwargs) -> List:
    """
    Create point-cloud splats for the Viser backend.

    Viser renders these as browser-native point clouds. The API mirrors the
    VisPy ``create_splats`` helper where possible, but per-point size and true
    Gaussian blending are backend-specific and are not available here.
    """
    pos = np.asarray(pos, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] not in (3, 4, 6, 7):
        raise ValueError("pos must have shape (N,3), (N,4), (N,6), or (N,7)")
    if len(pos) == 0:
        return []

    preset = dict(_POINT_CLOUD_MODE_PRESETS[_point_cloud_mode(mode)])
    if size is not None:
        preset['size'] = size
    if point_shape is not None:
        preset['point_shape'] = point_shape
    if point_shading is not None:
        preset['point_shading'] = point_shading
    if precision is not None:
        preset['precision'] = precision
    if np.asarray(preset['size']).ndim != 0:
        raise ValueError("viserplot.create_splats only supports scalar size")

    if values is not None:
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (len(pos), ):
            raise ValueError("values must have shape (N,)")

    colors = None
    color_arg = color
    if values is None and color is None:
        if pos.shape[1] == 4:
            values = pos[:, 3]
        elif pos.shape[1] in (6, 7):
            colors = pos[:, 3:]

    if color is not None:
        color_arr = np.asarray(color)
        if color_arr.ndim == 2:
            if color_arr.shape[0] != len(pos) or color_arr.shape[1] not in (3, 4):
                raise ValueError("per-point color must have shape (N,3) or (N,4)")
            colors = color_arr
            color_arg = None

    pos, values, colors, size = _sample_point_cloud_inputs(
        pos, values, colors, preset['size'], max_points, seed)

    if values is not None and colors is None and color_arg is None:
        pts = np.column_stack([pos[:, :3], values])
    else:
        pts = pos[:, :3]

    return create_points(
        pts,
        r=size,
        color=color_arg,
        cmap=cmap,
        clim=clim,
        values=values,
        colors=colors,
        point_shape=preset['point_shape'],
        point_shading=preset['point_shading'],
        precision=preset['precision'],
        **kwargs,
    )


_GAUSSIAN_SPLAT_MODE_PRESETS = {
    'point': {
        'radius': 2.0,
        'opacity': 0.75,
    },
    'surface': {
        'radius': (4.0, 4.0, 0.8),
        'opacity': 0.45,
    },
    'volume': {
        'radius': 2.5,
        'opacity': 0.28,
    },
}


def _sample_gaussian_splat_inputs(pos, values, colors, radius, covariances,
                                  opacity, max_points, seed):
    if max_points is None or len(pos) <= max_points:
        return pos, values, colors, radius, covariances, opacity

    max_points = int(max_points)
    if max_points <= 0:
        raise ValueError("max_points must be positive")

    n = len(pos)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, max_points, replace=False)
    pos = pos[idx]

    if values is not None:
        values = np.asarray(values)[idx]
    if colors is not None:
        colors = np.asarray(colors)[idx]
    if covariances is not None:
        covariances = np.asarray(covariances)[idx]

    radius_arr = np.asarray(radius)
    if radius_arr.shape == (n, 3) or (radius_arr.shape == (n, ) and n != 3):
        radius = radius_arr[idx]

    opacity_arr = np.asarray(opacity)
    if opacity_arr.ndim > 0 and opacity_arr.shape[0] == n:
        opacity = opacity_arr[idx]

    return pos, values, colors, radius, covariances, opacity


def _rgb_float(color, n):
    if color is None:
        return None

    arr = np.asarray(color)
    if arr.ndim == 2:
        if arr.shape != (n, 3) and arr.shape != (n, 4):
            raise ValueError("per-point color must have shape (N,3) or (N,4)")
        rgbs = arr[:, :3].astype(np.float32)
        if rgbs.size and rgbs.max() > 1:
            rgbs = rgbs / 255.0
        return np.clip(rgbs, 0.0, 1.0)

    if arr.ndim == 1 and arr.size in (3, 4):
        rgb = arr[:3].astype(np.float32)
        if rgb.size and rgb.max() > 1:
            rgb = rgb / 255.0
        return np.tile(np.clip(rgb, 0.0, 1.0), (n, 1))

    rgba = np.asarray(to_rgba(color), dtype=np.float32)
    return np.tile(rgba[:3], (n, 1))


def _gaussian_rgbs(pos, values, colors, color, cmap, clim):
    n = len(pos)
    rgbs = _rgb_float(color, n)
    if rgbs is not None:
        return rgbs.astype(np.float32)

    rgbs = _rgb_float(colors, n)
    if rgbs is not None:
        return rgbs.astype(np.float32)

    if values is None:
        values = pos[:, 2]
    values = np.asarray(values, dtype=np.float32)
    if clim is None:
        finite = np.isfinite(values)
        if np.any(finite):
            clim = [float(np.nanmin(values[finite])), float(np.nanmax(values[finite]))]
        else:
            clim = [0.0, 1.0]
    rgba = colormap.get_colors_from_cmap(cmap, clim, values)
    return np.asarray(rgba[:, :3], dtype=np.float32)


def _gaussian_opacities(opacity, n):
    arr = np.asarray(opacity, dtype=np.float32)
    if arr.ndim == 0:
        arr = np.full((n, 1), float(arr), dtype=np.float32)
    elif arr.shape == (n, ):
        arr = arr[:, None]
    elif arr.shape != (n, 1):
        raise ValueError("opacity must be a scalar, (N,), or (N,1)")
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _gaussian_covariances(radius, covariances, n):
    if covariances is not None:
        covariances = np.asarray(covariances, dtype=np.float32)
        if covariances.shape != (n, 3, 3):
            raise ValueError("covariances must have shape (N,3,3)")
        return covariances

    radius = np.asarray(radius, dtype=np.float32)
    if radius.ndim == 0:
        radii = np.full((n, 3), float(radius), dtype=np.float32)
    elif radius.shape == (3, ):
        radii = np.tile(radius, (n, 1)).astype(np.float32)
    elif radius.shape == (n, ):
        radii = np.repeat(radius[:, None], 3, axis=1).astype(np.float32)
    elif radius.shape == (n, 3):
        radii = radius.astype(np.float32)
    else:
        raise ValueError("radius must be scalar, (3,), (N,), or (N,3)")

    cov = np.zeros((n, 3, 3), dtype=np.float32)
    idx = np.arange(3)
    cov[:, idx, idx] = np.square(np.clip(radii, 1e-6, None))
    return cov


def create_gaussian_splats(pos: np.ndarray,
                           values: np.ndarray = None,
                           cmap: str = 'viridis',
                           clim: List = None,
                           color=None,
                           radius=None,
                           covariances=None,
                           opacity=None,
                           mode: str = 'surface',
                           max_points: int = None,
                           seed: int = 0,
                           scale=-1) -> List:
    """
    Create true Gaussian splats for the Viser backend.

    This wraps ``viser.SceneApi.add_gaussian_splats``. ``radius`` is expressed
    in the input data coordinate system before CIGVis scene scaling.
    """
    pos = np.asarray(pos, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] not in (3, 4, 6, 7):
        raise ValueError("pos must have shape (N,3), (N,4), (N,6), or (N,7)")
    if len(pos) == 0:
        return []

    preset = dict(_GAUSSIAN_SPLAT_MODE_PRESETS[_point_cloud_mode(mode)])
    if radius is not None:
        preset['radius'] = radius
    if opacity is not None:
        preset['opacity'] = opacity

    if values is not None:
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (len(pos), ):
            raise ValueError("values must have shape (N,)")

    colors = None
    color_arg = color
    if values is None and color is None:
        if pos.shape[1] == 4:
            values = pos[:, 3]
        elif pos.shape[1] in (6, 7):
            colors = pos[:, 3:]

    if color is not None:
        color_arr = np.asarray(color)
        if color_arr.ndim == 2:
            if color_arr.shape[0] != len(pos) or color_arr.shape[1] not in (3, 4):
                raise ValueError("per-point color must have shape (N,3) or (N,4)")
            colors = color_arr
            color_arg = None

    pos, values, colors, radius, covariances, opacity = _sample_gaussian_splat_inputs(
        pos,
        values,
        colors,
        preset['radius'],
        covariances,
        preset['opacity'],
        max_points,
        seed,
    )
    centers = pos[:, :3].astype(np.float32)
    n = len(centers)
    rgbs = _gaussian_rgbs(centers, values, colors, color_arg, cmap, clim)
    opacities = _gaussian_opacities(opacity, n)
    covariances = _gaussian_covariances(radius, covariances, n)

    return [
        GaussianSplatNode(
            centers,
            covariances,
            rgbs,
            opacities,
            scale=scale,
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
