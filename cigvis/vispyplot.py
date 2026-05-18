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

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable, List, Tuple, Dict, Union
import warnings
import os
import numpy as np
from cigvis.vispynodes import (
    VisCanvas,
    AxisAlignedImage,
    Colorbar,
    WellLog,
    XYZAxis,
    SurfaceNode,
    ArbLineNode,
    Axis3D,
    NorthPointer,
    Splat,
)
from cigvis.vispynodes.shading_filter import HeadlightShadingFilter

from vispy.color import ColorArray
from vispy.scene.visuals import Mesh, Line, Markers
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
    "create_point_cloud",
    "create_splats",
    "create_fault_skin",
    "create_arbitrary_line",
    "create_axis",
    "Plot3DView",
    "Plot3DSave",
    "Plot3DColorbar",
    "Plot3DGui",
    "plot3D",
    "run",
]


def _set_cbar_source(cbar, select: str, idx: int = 0, idx2: int = 0) -> None:
    source = {
        'select': select,
        'idx': int(idx),
        'idx2': int(idx2),
    }
    did_unfreeze = False
    if hasattr(cbar, 'unfreeze'):
        try:
            cbar.unfreeze()
            did_unfreeze = True
        except Exception:
            did_unfreeze = False
    try:
        cbar._cigvis_cbar_source = source
    finally:
        if did_unfreeze and hasattr(cbar, 'freeze'):
            cbar.freeze()


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
    shortcut_save_kw: Dict = None


@dataclass
class Plot3DSave:
    """Options that control automatic screenshots in ``plot3D``."""

    path: Any = None
    directory: Any = None
    transparent_bg: bool = True
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
    preserve_alpha: bool = None


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


def _flatten_nodes(nodes):
    if nodes is None:
        return []
    if isinstance(nodes, dict):
        out = []
        for value in nodes.values():
            out.extend(_flatten_nodes(value))
        return out
    if isinstance(nodes, (list, tuple)):
        out = []
        for item in nodes:
            out.extend(_flatten_nodes(item))
        return out
    return [nodes]


def _find_volume_images(nodes):
    managers = []
    for node in _flatten_nodes(nodes):
        vi = getattr(node, "_cigvis_volume_image", None)
        if vi is None:
            continue
        if not any(vi is manager for manager in managers):
            managers.append(vi)
    return managers


def _next_overlay_name(vi: VolumeImage) -> str:
    idx = len(getattr(vi, '_overlays', {}))
    while f'mask_{idx}' in vi._overlays:
        idx += 1
    return f'mask_{idx}'


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


def _cmap_name(cmap):
    return cmap if isinstance(cmap, str) else getattr(cmap, 'name', None)


_SPLAT_MODE_PRESETS = {
    'point': {
        'scaling': 'fixed',
        'size': 10.0,
        'alpha': 0.95,
        'sigma_rel': 0.42,
        'cutoff': 5e-3,
        'canvas_size_limits': None,
        'depth_mask': True,
    },
    'surface': {
        'scaling': 'visual',
        'size': 4.0,
        'alpha': 0.8,
        'sigma_rel': 0.50,
        'cutoff': 2e-3,
        'canvas_size_limits': (1.5, 14),
        'depth_mask': True,
    },
    'volume': {
        'scaling': 'visual',
        'size': 3.0,
        'alpha': 0.45,
        'sigma_rel': 0.58,
        'cutoff': 1e-3,
        'canvas_size_limits': (1.0, 18),
        'depth_mask': False,
    },
}


def _splat_mode(mode: str) -> str:
    aliases = {
        'point': 'point',
        'points': 'point',
        'pick': 'point',
        'picks': 'point',
        'surface': 'surface',
        'surf': 'surface',
        'volume': 'volume',
        'voxel': 'volume',
        'voxels': 'volume',
    }
    key = aliases.get(str(mode).lower())
    if key is None:
        raise ValueError("mode must be 'point', 'surface', or 'volume'")
    return key


def _normalize_splat_values(values, clim=None):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("values must be a 1D array")

    finite = np.isfinite(values)
    if clim is None:
        if np.any(finite):
            vmin = float(np.nanmin(values[finite]))
            vmax = float(np.nanmax(values[finite]))
        else:
            vmin, vmax = 0.0, 1.0
    else:
        if len(clim) != 2:
            raise ValueError("clim must contain two values")
        vmin, vmax = float(clim[0]), float(clim[1])

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0

    norm = np.zeros(values.shape, dtype=np.float32)
    norm[finite] = np.clip((values[finite] - vmin) / (vmax - vmin), 0.0, 1.0)
    return norm, (vmin, vmax)


def _splat_color_from_values(values, cmap, clim):
    norm, _ = _normalize_splat_values(values, clim)
    cmap = colormap.cmap_to_vispy(cmap)
    rgba = cmap.map(norm).astype(np.float32)
    rgba[:, 3] = 1.0
    return rgba


def _sample_splat_inputs(pos, values, color, size, max_points, seed):
    if max_points is None or len(pos) <= max_points:
        return pos, values, color, size

    n = len(pos)
    max_points = int(max_points)
    if max_points <= 0:
        raise ValueError("max_points must be positive")

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pos), max_points, replace=False)
    pos = pos[idx]

    if values is not None:
        values = np.asarray(values)[idx]

    size_arr = np.asarray(size)
    if size_arr.ndim == 1 and size_arr.shape == (n, ):
        size = size_arr[idx]

    color_arr = np.asarray(color) if color is not None else None
    if color_arr is not None and color_arr.ndim == 2 and color_arr.shape[0] == n:
        color = color_arr[idx]

    return pos, values, color, size


def create_splats(pos: np.ndarray,
                  values: np.ndarray = None,
                  cmap: str = 'viridis',
                  clim: List = None,
                  color=None,
                  size=None,
                  mode: str = 'surface',
                  scaling: str = None,
                  alpha: float = None,
                  sigma_rel: float = None,
                  cutoff: float = None,
                  antialias: float = 1.0,
                  canvas_size_limits: Tuple[float, float] = None,
                  max_points: int = None,
                  seed: int = 0,
                  premultiply: bool = True,
                  depth_test: bool = True,
                  depth_mask: bool = None,
                  **kwargs) -> List:
    """
    Create a Gaussian splat node from point positions.

    This is a user-facing wrapper around :class:`cigvis.vispynodes.Splat`.
    ``mode`` chooses practical defaults so callers usually only need
    positions plus either ``values``/``cmap`` or a fixed ``color``.

    Parameters
    ----------
    pos : array-like
        Point positions, shape ``(N, 2)`` or ``(N, 3)``.
    values : array-like, optional
        Per-point scalar values mapped through ``cmap``.
    cmap, clim
        Colormap and limits used when ``values`` is provided.
    color : color or array-like, optional
        Fixed or per-point RGBA colors. If supplied, it overrides
        ``values``/``cmap`` coloring.
    size : float or array-like, optional
        Splat diameter. Defaults depend on ``mode``.
    mode : {'point', 'surface', 'volume'}
        Preset for common use cases.
    max_points : int, optional
        Randomly subsample points before upload.

    Returns
    -------
    nodes : List
        A one-item list containing the Splat node, or an empty list when no
        points are provided.
    """
    pos = np.asarray(pos, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] not in (2, 3):
        raise ValueError("pos must have shape (N,2) or (N,3)")
    if len(pos) == 0:
        return []

    preset = dict(_SPLAT_MODE_PRESETS[_splat_mode(mode)])
    if scaling is not None:
        preset['scaling'] = scaling
    if size is not None:
        preset['size'] = size
    if alpha is not None:
        preset['alpha'] = alpha
    if sigma_rel is not None:
        preset['sigma_rel'] = sigma_rel
    if cutoff is not None:
        preset['cutoff'] = cutoff
    if canvas_size_limits is not None:
        preset['canvas_size_limits'] = canvas_size_limits
    if depth_mask is not None:
        preset['depth_mask'] = depth_mask

    if values is not None:
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (len(pos), ):
            raise ValueError("values must have shape (N,)")

    pos, values, color, size_data = _sample_splat_inputs(
        pos, values, color, preset['size'], max_points, seed)

    if color is None:
        if values is None:
            color = (1.0, 0.72, 0.18, 1.0)
        else:
            color = _splat_color_from_values(values, cmap, clim)

    splat = Splat(
        scaling=preset['scaling'],
        alpha=preset['alpha'],
        antialias=antialias,
        sigma_rel=preset['sigma_rel'],
        cutoff=preset['cutoff'],
        premultiply=premultiply,
        canvas_size_limits=preset['canvas_size_limits'],
        depth_test=depth_test,
        depth_mask=preset['depth_mask'],
        **kwargs,
    )
    splat.set_data(pos=pos, size=size_data, color=color)
    return [splat]


def create_slices(volume: np.ndarray,
                  pos: Union[List, Dict] = None,
                  clim: List = None,
                  cmap: str = 'Petrel',
                  interpolation: str = 'cubic',
                  texture_format=None,
                  display_range: Dict = None,
                  intersection_lines: bool = True,
                  line_color=(1, 1, 1),
                  line_width=2.0,
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
        colormap, it can be str or matplotlib's Colormap or vispy's Colormap
    interpolation : str
        interpolation method. If the values of the volume is discrete, we recommand 
        set as 'nearest'
    texture_format : None or 'auto',
        if use None, the NaNs will be clip to clim[1],
        and if use 'auto', the NaNs will be discarded, i.e., transparent
    display_range : Dict
        optional display ranges in original data coordinates, such as
        ``{'z': (0, 900)}``. Values are Python half-open ranges ``[start, stop)``.
    
    line_color : Tuple
        color for intersection lines and border lines, default is white
    line_width : float
        width for intersection lines and border lines, default is 2.0

    kwargs : Dict
        internal slice provider options

    Returns
    -------
    slices_nodes : List
        list of slice nodes
    """
    vi = VolumeImage(
        volume,
        cmap=cmap,
        clim=clim,
        interpolation=interpolation,
        texture_format=texture_format,
        display_range=display_range,
        **kwargs,
    )
    vi.create_slices(pos=pos)
    return vi.nodes(
        intersection_lines=intersection_lines,
        line_color=line_color,
        line_width=line_width,
    )


def add_mask(nodes: List,
             volume: Union[List, np.ndarray],
             clim: Union[List, Tuple] = None,
             cmap: Union[str, Dict] = None,
             interpolation: str = 'linear',
             alpha=None,
             excpt=None,
             method: str = 'auto',
             texture_format: str = 'auto',
             preproc_func: Callable = None,
             *,
             clims: Union[List, Tuple] = None,
             cmaps: Union[str, Dict] = None,
             preproc_funcs: Callable = None,
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
    if preproc_funcs is not None:
        if preproc_func is not None:
            raise ValueError("Specify only one of 'preproc_func' or 'preproc_funcs'")
        preproc_func = preproc_funcs

    if _is_volume_sequence(volume):
        if len(volume) != 1:
            raise ValueError(
                "add_mask no longer accepts multiple volumes. "
                "Call add_mask repeatedly, for example: "
                "nodes = cigvis.add_mask(nodes, rgt, cmap='stratum'); "
                "nodes = cigvis.add_mask(nodes, fault, cmap='jet')"
            )
        volume = volume[0]

    interpolation = _normalize_single_value(interpolation, "interpolation")
    alpha = _normalize_single_value(alpha, "alpha")
    excpt = _normalize_single_value(excpt, "excpt")
    preproc_func = _normalize_single_value(preproc_func, "preproc_func")
    cmap = _normalize_cmap_value(cmap)
    clim = _normalize_clim_value(clim)

    utils.check_mmap(volume)
    if cmap is None:
        raise ValueError("'cmap' cannot be None")

    def _prepare_cmap(cmap_value):
        name = _cmap_name(cmap_value)
        if alpha is not None:
            cmap_value = colormap.fast_set_cmap(cmap_value, alpha, excpt)
        return cmap_value, name

    if isinstance(cmap, dict):
        cmap_for_manager = {}
        cmap_names = {}
        for axis, cmap_value in cmap.items():
            axis = str(axis).lower()
            if axis not in ('x', 'y', 'z'):
                raise ValueError("cmap dict keys must be one of 'x', 'y', or 'z'")
            cmap_value, name = _prepare_cmap(cmap_value)
            cmap_for_manager[axis] = cmap_value
            cmap_names[axis] = name
        axes = tuple(cmap_for_manager.keys())
    else:
        cmap_for_manager, cmap_names = _prepare_cmap(cmap)
        axes = None

    managers = _find_volume_images(nodes)
    if len(managers) > 1:
        raise ValueError(
            "add_mask found slices from multiple VolumeImage managers. "
            "Call add_mask separately for each create_slices result."
        )

    if len(managers) == 1:
        vi = managers[0]
        vi.add_overlay_volume(
            name=_next_overlay_name(vi),
            volume=volume,
            cmap=cmap_for_manager,
            cmap_names=cmap_names,
            clim=clim,
            interpolation=interpolation,
            preproc=preproc_func,
            method=method,
            texture_format=texture_format,
            axes=axes,
        )
        return nodes

    if clim is None:
        clim = utils.auto_clim(volume)

    flat_nodes = _flatten_nodes(nodes)
    for node in flat_nodes:
        if not isinstance(node, AxisAlignedImage):
            continue
        if isinstance(cmap_for_manager, dict):
            if node.axis not in cmap_for_manager:
                continue
            node_cmap = colormap.cmap_to_vispy(cmap_for_manager[node.axis])
            node_cmap_name = cmap_names[node.axis]
        else:
            node_cmap = colormap.cmap_to_vispy(cmap_for_manager)
            node_cmap_name = cmap_names
        node.add_mask(
            volume,
            node_cmap,
            clim,
            interpolation,
            method=method,
            texture_format=texture_format,
            preproc_f=preproc_func,
        )
        image = node.overlaid_images[-1]
        _set_visual_metadata(
            image,
            _cigvis_cmap_name=node_cmap_name,
            _cigvis_interpolation=interpolation,
        )

    return nodes


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
    Deprecated overlay API.

    Use ``create_slices(bg_volume)`` followed by ``add_mask(...)``.
    """
    raise RuntimeError(
        "cigvis.create_overlay is no longer supported for the VisPy backend. "
        "Use `nodes = cigvis.create_slices(bg_volume)` and then "
        "`nodes = cigvis.add_mask(nodes, fg_volume, cmap='jet')`."
    )


def create_colorbar(cmap,
                    clim: List,
                    discrete: bool = False,
                    disc_ticks: Union[List, Dict] = None,
                    label_str: str = '',
                    preserve_alpha: bool = False,
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
    preserve_alpha : bool, optional
        Keep cmap alpha in the colorbar if True. The default draws colorbar
        colors opaque.
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
                    preserve_alpha=preserve_alpha,
                    **kwargs)

    return cbar


def create_colorbar_from_nodes(nodes,
                               label_str='',
                               select='auto',
                               idx=0,
                               idx2=0,
                               preserve_alpha: bool = False,
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
    preserve_alpha : bool, optional
        Keep cmap alpha in the colorbar if True. The default draws colorbar
        colors opaque.
    kwargs : Dict
        Other params for Colorbar.
    """
    # fmt: off
    assert len(nodes) > 0, "there is no node, len(nodes) == 0"
    source_select = select
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
    source_select = select

    assert select in ['last', 'mask', 'surface', 'slices', 'logs', 'line_logs', 'mesh']
    cmap = None
    clim = None
    if select == 'mask' or (select == 'last' and isinstance(nodes[-1], AxisAlignedImage) and len(nodes[-1].overlaid_images) > 1):
        source_select = 'mask'
        if select != 'last':
            node = [
                node for node in nodes
                if isinstance(node, AxisAlignedImage)
                and len(node.overlaid_images) > 1
            ]
            if len(node) == 0:
                raise ValueError("No valid nodes, no AxisAlignedImage with mask")
            if len(node[0].overlaid_images) == 2:
                idx = 0
            elif len(node[0].overlaid_images) <= idx + 1:
                raise ValueError(f"idx error, there are only {len(node[0].overlaid_images)-1} mask, but got idx = {idx}")
        else:
            node = [nodes[-1]]
        cmap = node[0].overlaid_images[idx + 1].cmap
        clim = node[0].overlaid_images[idx + 1].clim
    elif select == 'surface' or (select == 'last' and isinstance(nodes[-1], SurfaceNode)):
        source_select = 'surface'
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
        source_select = 'slices'
        if select != 'last':
            node = [node for node in nodes if isinstance(node, AxisAlignedImage)]
            if len(node) == 0:
                raise ValueError("No valid nodes, no AxisAlignedImage")
        else:
            node = [nodes[-1]]
        cmap = node[0].overlaid_images[0].cmap
        clim = node[0].overlaid_images[0].clim
    elif select == 'logs' or (select == 'last' and isinstance(nodes[-1], WellLog)):
        source_select = 'logs'
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

    cbar = Colorbar(cmap=cmap,
                    clim=clim,
                    label_str=label_str,
                    preserve_alpha=preserve_alpha,
                    **kwargs)
    _set_cbar_source(cbar, source_select, idx, idx2)
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
        node.update_colors_by_slice_node([alignImage], volumes)

    return nodes


def create_surfaces(surfs: List[np.ndarray],
                    volume: np.ndarray = None,
                    value_type: str = 'depth',
                    clim: List = None,
                    cmap: str = 'jet',
                    shape: Union[Tuple, List] = None,
                    interp: bool = False,
                    quad: bool = False,
                    quad_size: Union[float, Tuple, List] = 1.0,
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
    quad : bool
        If True, treat each ``(N, 3)`` input point as a separate x-y aligned
        quad patch instead of interpolating/connecting points into a grid.
        This is useful for sparse or incomplete gentle horizons.
    quad_size : float or tuple
        Full side length of each quad patch when ``quad=True``. A scalar
        creates square patches; a two-element tuple controls x/y size.
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
                           quad_size=quad_size,
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


def _sample_point_cloud_inputs(pos, values, colors, max_points, seed):
    if max_points is None or len(pos) <= max_points:
        return pos, values, colors

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
    return pos, values, colors


def _point_cloud_colors(pos, values, cmap, clim, color, colors):
    n = len(pos)
    if color is not None:
        rgba = ColorArray(color).rgba
        if rgba.shape == (1, 4):
            rgba = np.tile(rgba, (n, 1))
        elif rgba.shape != (n, 4):
            raise ValueError("color must be a single color or an (N, 4) array")
        return rgba.astype(np.float32, copy=False)

    if colors is not None:
        rgba = np.asarray(colors, dtype=np.float32)
        if rgba.ndim != 2 or rgba.shape[0] != n or rgba.shape[1] not in (3, 4):
            raise ValueError("colors must have shape (N, 3) or (N, 4)")
        if rgba.shape[1] == 3:
            rgba = np.concatenate(
                [rgba, np.ones((n, 1), dtype=np.float32)], axis=1)
        else:
            rgba = rgba.copy()
        finite = rgba[np.isfinite(rgba)]
        if finite.size and finite.max() > 1.0:
            rgba /= 255.0
        return np.clip(rgba, 0.0, 1.0).astype(np.float32, copy=False)

    if values is None:
        values = pos[:, 2]
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if clim is None:
        if np.any(finite):
            clim = [float(np.nanmin(values[finite])),
                    float(np.nanmax(values[finite]))]
        else:
            clim = [0.0, 1.0]
    if clim[0] == clim[1]:
        clim = [clim[0], clim[0] + 1.0]
    return colormap.get_colors_from_cmap(cmap, clim,
                                         values).astype(np.float32, copy=False)


def create_point_cloud(points: np.ndarray,
                       values: np.ndarray = None,
                       cmap: str = 'viridis',
                       clim: List = None,
                       color=None,
                       colors: np.ndarray = None,
                       size: float = 4.0,
                       symbol: str = 'o',
                       edge_color=None,
                       edge_width: float = 0,
                       max_points: int = None,
                       seed: int = 0,
                       depth_test: bool = True,
                       depth_mask: bool = True,
                       blend: bool = False,
                       **kwargs):
    """
    Create a lightweight VisPy marker point-cloud node.

    Parameters
    ----------
    points : array-like
        Point positions. Shape can be ``(N, 3)``, ``(N, 4)`` with scalar
        values in the last column, or ``(N, 6)/(N, 7)`` with RGB/RGBA colors
        in the trailing columns.
    values : array-like, optional
        Per-point scalar values mapped by ``cmap`` and ``clim``.
    color, colors : optional
        A single color, or per-point RGB/RGBA colors.
    size : float
        Marker size in screen pixels.
    max_points : int, optional
        Randomly subsample points before upload.

    Returns
    -------
    nodes : List
        A one-item list containing a ``Markers`` node, or an empty list when
        no points are provided.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] not in (3, 4, 6, 7):
        raise ValueError("points must have shape (N,3), (N,4), (N,6), or (N,7)")
    if len(points) == 0:
        return []

    n = len(points)
    pos = np.ascontiguousarray(points[:, :3], dtype=np.float32)
    if values is None and color is None and colors is None:
        if points.shape[1] == 4:
            values = points[:, 3]
        elif points.shape[1] in (6, 7):
            colors = points[:, 3:]

    if values is not None:
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (n, ):
            raise ValueError("values must have shape (N,)")

    if color is not None:
        color_arr = np.asarray(color)
        if color_arr.ndim == 2:
            colors = color_arr
            color = None

    pos, values, colors = _sample_point_cloud_inputs(pos, values, colors,
                                                     max_points, seed)
    face_color = _point_cloud_colors(pos, values, cmap, clim, color, colors)
    if edge_color is None:
        edge_color = face_color

    marker = Markers(**kwargs)
    marker.set_data(pos=pos,
                    size=float(size),
                    face_color=face_color,
                    edge_color=edge_color,
                    edge_width=edge_width,
                    symbol=symbol)
    marker.set_gl_state(depth_test=depth_test,
                        depth_mask=depth_mask,
                        blend=blend)
    return [marker]


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
        'shortcut_save_kw',
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
    ...         transparent_bg=True,
    ...     ),
    ...     cbar=cigvis.Plot3DColorbar(save=False),
    ...     gui=cigvis.Plot3DGui(enabled=True, theme='dark'),
    ... )

    Plain dicts are also accepted for ``view``/``save``/``cbar``/``gui``.
    Automatic PNG saves capture the visible canvas framebuffer, the same path
    used by the ``s`` keyboard shortcut.

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
    if 'shortcut_save_kw' not in canvas_kw and save_kw:
        canvas_kw = dict(canvas_kw)
        canvas_kw['shortcut_save_kw'] = dict(save_kw)

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
        from cigvis.gui.gui3d import launch_plot3d_gui

        gui_canvas_kw = dict(canvas_kw)
        gui_canvas_kw.update({
            'size': size,
            'cbar_region_ratio': cbar_region_ratio,
            'savedir': save_dir,
        })
        win = launch_plot3d_gui(
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
