# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
r"""
Functions for drawing 3D seismic figure using plotly
----------------------------------------------------

TODO: The code for the `plotly` part is not yet fully developed,
and there are only some basic implementations.
We will continue to improve it in the future.

Note
----
Only run in jupyter environment (not include Ipython)

In plotly, for a seismic volume,
- x means inline order
- y means crossline order
- z means time order

- ni means the dimension size of inline / x
- nx means the dimension size of crossline / y
- nt means the dimension size of time / depth / z


Examples
--------
>>> volume.shape = (192, 200, 240) # (ni, nx, nt)

\# only slices
>>> nodes = create_slices(volume, pos=[0, 0, 239], cmap='Petrel', show_cbar=True)
>>> plot3D(nodes)

\# add mask
>>> nodes = create_slices(volume, pos=[0, 0, 239], cmap='gray')
>>> nodes = add_mask(nodes, mask, cmap='jet')
>>> plot3D(nodes)

\# add surface

\# surfs = [surf1, surf2, ...], each shape is (ni, nx)
>>> sf_nodes = create_surfaces(surfs, value_type='depth')

\# or use amplitude as color
>>> sf_nodes = create_surfaces(surfs, volume, value_type='amp')
>>> plot3D(nodes + sf_nodes)

For more and detail examples, please refer our documents
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Union
import copy
import warnings

import numpy as np
from cigvis import ExceptionWrapper

try:
    import plotly.graph_objects as go
    from plotly.basedatatypes import BaseTraceType
except BaseException as e:
    go = ExceptionWrapper(
        e,
        "run `pip install \"cigvis[plotly]\"` or run `pip install \"cigvis[all]\"` to enable jupyter support"
    )
    BaseTraceType = ()

from skimage.measure import marching_cubes
from skimage import transform

import cigvis
from cigvis import colormap
from cigvis.utils import plotlyutils
from cigvis.utils.slice_provider import SliceProvider
import cigvis.utils as utils


__all__ = [
    "PlotlySpec",
    "PlotlySliceSpec",
    "PlotlyOverlaySpec",
    "PlotlySurfacesSpec",
    "PlotlyLineLogSpec",
    "PlotlyPointsSpec",
    "PlotlyBodySpec",
    "create_slices",
    "add_mask",
    "create_overlay",
    "create_surfaces",
    "create_line_logs",
    "create_well_logs",
    "create_points",
    "create_bodies",
    "create_fault_skin",
    "compile_traces",
    "plot3D",
]


_AXES = ("x", "y", "z")
_DIMNAME = dict(x="inline", y="crossline", z="time")
_INTERP_ORDER = dict(nearest=0, linear=1, bilinear=1, cubic=3)


class PlotlySpec:
    """Lightweight Plotly backend object compiled by ``plot3D``."""

    def to_traces(self) -> List:
        raise NotImplementedError


@dataclass
class PlotlyOverlaySpec(PlotlySpec):
    volume: np.ndarray
    cmap: Any
    clim: Tuple[float, float]
    interpolation: str = "linear"
    preproc_func: Callable = None
    provider: Any = None
    show_cbar: bool = False
    cbar_params: Dict = None
    nancolor: Any = None

    def to_traces(self) -> List:
        return []


@dataclass
class PlotlySliceSpec(PlotlySpec):
    volume: np.ndarray
    axis: str
    pos: int
    cmap: Any = "Petrel"
    clim: Tuple[float, float] = None
    scale: float = 1
    interpolation: str = "cubic"
    display_range: Dict = None
    provider: Any = None
    overlays: List[PlotlyOverlaySpec] = field(default_factory=list)
    show_cbar: bool = False
    cbar_params: Dict = None
    nancolor: Any = None
    name: str = None
    kwargs: Dict = field(default_factory=dict)

    def to_traces(self) -> List:
        return _compile_slice(self)


@dataclass
class PlotlySurfacesSpec(PlotlySpec):
    surfs: List[np.ndarray]
    volume: np.ndarray = None
    value_type: str = "depth"
    clim: Tuple[float, float] = None
    cmap: Any = "jet"
    show_cbar: bool = False
    kwargs: Dict = field(default_factory=dict)

    def to_traces(self) -> List:
        return _compile_surfaces(self)


@dataclass
class PlotlyLineLogSpec(PlotlySpec):
    log: np.ndarray
    cmap: Any = "jet"
    line_width: float = 8

    def to_traces(self) -> List:
        return [_compile_line_log(self)]


@dataclass
class PlotlyPointsSpec(PlotlySpec):
    points: np.ndarray
    color: Any = "red"
    size: float = 3
    sym: str = "square"

    def to_traces(self) -> List:
        return [_compile_points(self)]


@dataclass
class PlotlyBodySpec(PlotlySpec):
    volume: np.ndarray
    level: float
    margin: float = None
    color: Any = "yellow"

    def to_traces(self) -> List:
        return [_compile_body(self)]


def _is_volume_sequence(value) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    if len(value) == 0:
        return False
    return all(hasattr(item, "ndim") or type(item).__module__ == "torch" for item in value)


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


def _normalize_axis(axis: str) -> str:
    axis = str(axis).lower()
    aliases = dict(inline="x", crossline="y", time="z", depth="z")
    axis = aliases.get(axis, axis)
    if axis not in _AXES:
        raise ValueError("axis must be one of 'x', 'y', or 'z'")
    return axis


def _as_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _logical_shape(volume) -> Tuple[int, int, int]:
    if isinstance(volume, SliceProvider):
        return tuple(volume.shape)
    if isinstance(volume, dict):
        return tuple(SliceProvider(volume).shape)
    shape, _ = utils.get_shape(volume, cigvis.is_line_first())
    return tuple(shape)


def _plotly_shape(volume) -> Tuple[int, int, int]:
    shape = _logical_shape(volume)
    if cigvis.is_line_first():
        return shape
    return tuple(shape[::-1])


def _normalize_pos(pos, volume) -> Dict[str, List[int]]:
    if pos is None:
        nt = _logical_shape(volume)[2]
        pos = dict(x=[0], y=[0], z=[nt - 1])

    if isinstance(pos, (list, tuple)):
        if len(pos) != 3:
            raise ValueError("pos list must contain three entries for x/y/z")
        if any(isinstance(v, (list, tuple, np.ndarray)) for v in pos):
            x, y, z = pos
        else:
            x, y, z = [pos[0]], [pos[1]], [pos[2]]
        pos = dict(x=x, y=y, z=z)

    if not isinstance(pos, dict):
        raise TypeError("pos must be None, a list/tuple, or a dict")

    out = {axis: [] for axis in _AXES}
    for axis, values in pos.items():
        axis = _normalize_axis(axis)
        out[axis] = [int(v) for v in _as_list(values)]
    return out


def _normalize_sequence(value, nitems, name, allow_single=True):
    if allow_single and not isinstance(value, (list, tuple)):
        return [value] * nitems
    if allow_single and name.endswith("cmap") and isinstance(value, str):
        return [value] * nitems
    if isinstance(value, (list, tuple)) and len(value) == nitems:
        return list(value)
    if allow_single and nitems == 1:
        return [value]
    raise ValueError(f"{name} must contain {nitems} item(s)")


def _is_clim_pair(value) -> bool:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    return not any(isinstance(item, (list, tuple, np.ndarray)) for item in value)


def _normalize_clim_sequence(value, volumes, name):
    if value is None:
        return [utils.auto_clim(volume) for volume in volumes]
    if _is_clim_pair(value):
        return [value] * len(volumes)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == len(volumes):
        return list(value)
    if len(volumes) == 1:
        return [value]
    raise ValueError(f"{name} must contain {len(volumes)} clim item(s)")


def _prepare_cmap(cmap, alpha=None, excpt=None):
    if alpha is not None:
        cmap = colormap.fast_set_cmap(cmap, alpha, excpt)
    return cmap


def _interp_order(interpolation) -> int:
    if interpolation is None:
        return 3
    if isinstance(interpolation, str):
        return _INTERP_ORDER.get(interpolation.lower(), 3)
    return int(interpolation)


def _resize_slice(image, scale=1, interpolation="cubic"):
    if scale is None or scale == 1:
        return image
    if scale <= 0:
        raise ValueError("scale must be positive")

    n0 = max(1, int(round(image.shape[0] / scale)))
    n1 = max(1, int(round(image.shape[1] / scale)))
    order = _interp_order(interpolation)
    anti_aliasing = order > 0

    if image.ndim == 2:
        target_shape = (n0, n1)
    else:
        target_shape = (n0, n1) + tuple(image.shape[2:])

    return transform.resize(
        image,
        target_shape,
        order=order,
        anti_aliasing=anti_aliasing,
        preserve_range=True,
    )


def _slice_image(volume, axis, pos, preproc_func=None, provider=None):
    if provider is not None:
        return provider(axis, pos)
    image = plotlyutils.get_image_func(volume, axis, pos, preproc_func)
    return image


def _rgba_to_plotly(colors) -> List[str]:
    out = []
    colors = np.asarray(colors)
    for color in colors:
        rgb = np.clip(color[:3] * 255, 0, 255).astype(int)
        alpha = float(np.clip(color[3], 0, 1))
        out.append(f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha:.6g})")
    return out


def _make_colorbar_trace(cmap, clim, cbar_params=None, name=None):
    colorbar = copy.deepcopy(cbar_params) if cbar_params is not None else {}
    return go.Scatter3d(
        x=[None],
        y=[None],
        z=[None],
        mode="markers",
        marker=dict(
            size=0.1,
            color=[clim[0]],
            colorscale=colormap.cmap_to_plotly(cmap),
            cmin=clim[0],
            cmax=clim[1],
            colorbar=colorbar,
            showscale=True,
        ),
        name=name,
        showlegend=False,
    )


def _compile_slice(spec: PlotlySliceSpec) -> List:
    source = spec.provider or spec.volume
    shape = _plotly_shape(source)
    base = _slice_image(spec.volume,
                        spec.axis,
                        spec.pos,
                        provider=spec.provider)
    base = _resize_slice(base, spec.scale, spec.interpolation)
    name = spec.name or f"{spec.axis}/{_DIMNAME[spec.axis]}"

    xx, yy, zz = plotlyutils.make_xyz(spec.pos, shape, spec.axis, base.shape[:2])

    if len(spec.overlays) == 0:
        kwargs = dict(
            x=xx,
            y=yy,
            z=zz,
            surfacecolor=base,
            colorscale=colormap.cmap_to_plotly(spec.cmap),
            cmin=spec.clim[0],
            cmax=spec.clim[1],
            name=name,
            colorbar=spec.cbar_params,
            showscale=spec.show_cbar,
            showlegend=False,
        )
        kwargs.update(spec.kwargs)
        return [go.Surface(**kwargs)]

    arrays = [base]
    cmaps = [spec.cmap]
    clims = [spec.clim]
    nancolor = spec.nancolor

    for overlay in spec.overlays:
        image = _slice_image(
            overlay.volume,
            spec.axis,
            spec.pos,
            overlay.preproc_func,
            provider=overlay.provider,
        )
        image = _resize_slice(image, spec.scale, overlay.interpolation)
        if image.shape[:2] != base.shape[:2]:
            image = transform.resize(
                image,
                base.shape[:2],
                order=_interp_order(overlay.interpolation),
                anti_aliasing=_interp_order(overlay.interpolation) > 0,
                preserve_range=True,
            )
        arrays.append(image)
        cmaps.append(overlay.cmap)
        clims.append(overlay.clim)
        if overlay.nancolor is not None:
            nancolor = overlay.nancolor

    colors = colormap.arrs_to_image(
        arrays,
        cmaps,
        clims,
        as_uint8=False,
        nancolor=nancolor,
    ).reshape(-1, 4)
    vertexcolor = _rgba_to_plotly(colors)

    x, y, z, ii, jj, kk = plotlyutils.make_triang(xx, yy, zz)
    traces = [
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=ii,
            j=jj,
            k=kk,
            name=name,
            vertexcolor=vertexcolor,
            showlegend=False,
        )
    ]

    if spec.show_cbar:
        traces.append(
            _make_colorbar_trace(
                spec.cmap,
                spec.clim,
                spec.cbar_params,
                name=f"{name} colorbar",
            ))

    for overlay in spec.overlays:
        if overlay.show_cbar:
            traces.append(
                _make_colorbar_trace(
                    overlay.cmap,
                    overlay.clim,
                    overlay.cbar_params,
                    name=f"{name} overlay colorbar",
                ))

    return traces


def _compile_surfaces(spec: PlotlySurfacesSpec) -> List:
    line_first = cigvis.is_line_first()

    surfs = spec.surfs
    if spec.value_type == "amp":
        if spec.volume is None:
            raise ValueError("Must input volume if value_type is 'amp' (amplitude)")
        surf_values = [utils.surfaceutils.interp_surf(spec.volume, surf) for surf in surfs]
    else:
        surf_values = copy.deepcopy(surfs)

    if spec.clim is None:
        vmin = min([np.nanmin(value) for value in surf_values])
        vmax = max([np.nanmax(value) for value in surf_values])
    else:
        vmin, vmax = spec.clim

    traces = []
    for surf, value in zip(surfs, surf_values):
        if line_first:
            surf = surf.T
            value = value.T

        kwargs = dict(
            z=surf,
            surfacecolor=value,
            colorscale=colormap.cmap_to_plotly(spec.cmap),
            cmin=vmin,
            cmax=vmax,
            showscale=spec.show_cbar,
            lighting=dict(
                ambient=0.1,
                diffuse=0.9,
                specular=0.5,
                roughness=0.3,
                fresnel=0.5,
            ),
            lightposition=dict(x=100, y=200, z=300),
        )
        kwargs.update(spec.kwargs)
        traces.append(go.Surface(**kwargs))

    return traces


def _compile_line_log(spec: PlotlyLineLogSpec):
    log = np.asarray(spec.log)
    if log.shape[1] < 3:
        raise ValueError("log must have at least three columns")
    value = log[:, 2] if log.shape[1] == 3 else log[:, 3]
    return go.Scatter3d(
        x=log[:, 0],
        y=log[:, 1],
        z=log[:, 2],
        line=dict(
            color=value,
            colorscale=colormap.cmap_to_plotly(spec.cmap),
            width=spec.line_width,
        ),
        mode="lines",
        showlegend=False,
    )


def _compile_points(spec: PlotlyPointsSpec):
    points = np.asarray(spec.points)
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(
            symbol=spec.sym,
            size=spec.size,
            color=spec.color,
            line=dict(width=1, color="black"),
        ),
        showlegend=False,
    )


def _compile_body(spec: PlotlyBodySpec):
    volume = spec.volume
    if spec.margin is not None:
        volume = np.array(volume, copy=True)
        volume[0, :, :] = spec.margin
        volume[:, 0, :] = spec.margin
        volume[:, :, 0] = spec.margin
        volume[volume.shape[0] - 1, :, :] = spec.margin
        volume[:, volume.shape[1] - 1, :] = spec.margin
        volume[:, :, volume.shape[2] - 1] = spec.margin

    vertices, faces, _, _ = marching_cubes(volume, spec.level)
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=spec.color,
        showscale=False,
        flatshading=False,
        lighting=dict(
            ambient=0.1,
            diffuse=0.9,
            specular=0.5,
            roughness=0.3,
            fresnel=0.5,
        ),
        lightposition=dict(x=100, y=200, z=300),
    )


def _is_plotly_trace(value) -> bool:
    if BaseTraceType and isinstance(value, BaseTraceType):
        return True
    return isinstance(value, dict) and "type" in value


def _flatten_nodes(nodes):
    if nodes is None:
        return []
    if _is_plotly_trace(nodes) or isinstance(nodes, PlotlySpec):
        return [nodes]
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


def compile_traces(nodes) -> List:
    """Compile Plotly specs and raw Plotly traces into a flat trace list."""
    traces = []
    for node in _flatten_nodes(nodes):
        if isinstance(node, PlotlySpec):
            traces.extend(node.to_traces())
        elif _is_plotly_trace(node):
            traces.append(node)
        else:
            raise TypeError(
                "plot3D only accepts PlotlySpec objects, Plotly traces, "
                "or nested lists/dicts containing them"
            )
    return traces


def create_slices(volume: np.ndarray,
                  pos: Union[List, Dict] = None,
                  clim: List = None,
                  cmap: str = "Petrel",
                  scale: float = 1,
                  show_cbar: bool = False,
                  cbar_params: Dict = None,
                  interpolation: str = "cubic",
                  texture_format=None,
                  display_range: Dict = None,
                  nancolor=None,
                  **kwargs):
    """
    Create slice specs. Specs are materialized into Plotly traces by ``plot3D``.

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
    show_cbar : bool
        show colorbar
    cbar_params : Dict
        parameters pass to colorbar

    Returns
    -------
    specs : List
        List of PlotlySliceSpec
    """
    provider = SliceProvider(
        volume,
        display_range=display_range,
        transpose_line_first=True,
        transpose_rgb=True,
    )
    pos = _normalize_pos(pos, provider)
    if clim is None:
        clim = utils.auto_clim(provider.clim_source)

    specs = []
    idx = 0
    for axis in _AXES:
        for p in pos[axis]:
            specs.append(
                PlotlySliceSpec(
                    volume=volume,
                    axis=axis,
                    pos=p,
                    cmap=cmap,
                    clim=tuple(clim),
                    scale=scale,
                    interpolation=interpolation,
                    display_range=display_range,
                    provider=provider,
                    show_cbar=bool(show_cbar and idx == 0),
                    cbar_params=copy.deepcopy(cbar_params),
                    nancolor=nancolor,
                    kwargs=copy.deepcopy(kwargs),
                ))
            idx += 1
    return specs


def add_mask(nodes: List,
             volume: Union[List, np.ndarray],
             clim: Union[List, Tuple] = None,
             cmap: Union[str, Dict] = None,
             interpolation: str = "linear",
             alpha=None,
             excpt=None,
             method: str = "auto",
             texture_format: str = "auto",
             preproc_func: Callable = None,
             *,
             clims: Union[List, Tuple] = None,
             cmaps: Union[str, Dict] = None,
             preproc_funcs: Callable = None,
             show_cbar: bool = False,
             cbar_params: Dict = None,
             nancolor=None,
             **kwargs) -> List:
    """
    Add a mask/overlay volume to Plotly slice specs.

    Parameters
    -----------
    nodes: List[PlotlySliceSpec]
        A List that contains specs created by ``create_slices``.
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
        interpolation method. If the values of the slices is discrete, we
        recommend set as 'nearest'
    alpha : float or List[float]
        if alpha is not None, using `colormap.fast_set_cmap` to set cmap
    excpt : None or str
        it could be one of [None, 'min', 'max', 'ramp']

    Returns
    -------
    slices_nodes : List
        list of slice specs
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
                "nodes = plotlyplot.add_mask(nodes, rgt, cmap='stratum'); "
                "nodes = plotlyplot.add_mask(nodes, fault, cmap='jet')"
            )
        volume = volume[0]

    interpolation = _normalize_single_value(interpolation, "interpolation")
    alpha = _normalize_single_value(alpha, "alpha")
    excpt = _normalize_single_value(excpt, "excpt")
    preproc_func = _normalize_single_value(preproc_func, "preproc_func")
    cmap = _normalize_cmap_value(cmap)
    clim = _normalize_clim_value(clim)

    if cmap is None:
        raise ValueError("'cmap' cannot be None")
    base_specs = [
        node for node in _flatten_nodes(nodes)
        if isinstance(node, PlotlySliceSpec)
    ]
    display_range = None
    if base_specs:
        base_provider = base_specs[0].provider
        display_range = getattr(base_provider, 'display_range',
                                base_specs[0].display_range)
    provider = SliceProvider(
        volume,
        preproc=preproc_func,
        display_range=display_range,
        transpose_line_first=True,
        transpose_rgb=True,
    )
    if clim is None:
        clim = utils.auto_clim(provider.clim_source)

    if isinstance(cmap, dict):
        cmap_by_axis = {}
        for axis, cmap_value in cmap.items():
            axis = _normalize_axis(axis)
            cmap_by_axis[axis] = _prepare_cmap(cmap_value, alpha, excpt)
    else:
        cmap_by_axis = None
        cmap_value = _prepare_cmap(cmap, alpha, excpt)

    added = 0
    for node in _flatten_nodes(nodes):
        if not isinstance(node, PlotlySliceSpec):
            continue
        if _logical_shape(node.provider or node.volume) != provider.shape:
            raise ValueError("mask volume shape must match the base slice volume shape")
        if cmap_by_axis is not None:
            if node.axis not in cmap_by_axis:
                continue
            node_cmap = cmap_by_axis[node.axis]
        else:
            node_cmap = cmap_value
        node.overlays.append(
            PlotlyOverlaySpec(
                volume=volume,
                cmap=node_cmap,
                clim=tuple(clim),
                interpolation=interpolation,
                preproc_func=preproc_func,
                provider=provider,
                show_cbar=bool(show_cbar and added == 0),
                cbar_params=copy.deepcopy(cbar_params),
                nancolor=nancolor,
            ))
        added += 1

    return nodes


def create_overlay(bg_volume: np.ndarray,
                   fg_volume: np.ndarray,
                   pos: Union[List, Dict] = None,
                   bg_clim: List = None,
                   fg_clim: List = None,
                   bg_cmap: str = "Petrel",
                   fg_cmap: str = None,
                   show_cbar: bool = False,
                   cbar_type: str = "fg",
                   bg_interpolation: str = "cubic",
                   fg_interpolation: str = "linear",
                   **kwargs):
    """
    Deprecated-style overlay helper.

    New code should prefer ``create_slices(bg_volume)`` followed by
    ``add_mask(...)``. This helper now builds the same Plotly specs.
    """
    utils.check_mmap(bg_volume)
    fg_volumes = fg_volume if _is_volume_sequence(fg_volume) else [fg_volume]
    for volume in fg_volumes:
        if _logical_shape(bg_volume) != _logical_shape(volume):
            raise ValueError("foreground volume shape must match background volume shape")
        utils.check_mmap(volume)

    if fg_cmap is None:
        raise ValueError("'fg_cmap' cannot be None")

    cbar_type = str(cbar_type).lower()
    if cbar_type not in ("bg", "base", "fg", "foreground"):
        raise ValueError("cbar_type must be 'bg'/'base' or 'fg'/'foreground'")

    scale = kwargs.pop("scale", 1)
    cbar_params = kwargs.pop("cbar_params", None)
    alpha = kwargs.pop("alpha", None)
    excpt = kwargs.pop("excpt", None)
    preproc_func = kwargs.pop("preproc_func", None)
    nancolor = kwargs.pop("nancolor", None)

    fg_clims = _normalize_clim_sequence(fg_clim, fg_volumes, "fg_clim")
    fg_cmaps = _normalize_sequence(fg_cmap, len(fg_volumes), "fg_cmap")
    fg_interps = _normalize_sequence(fg_interpolation, len(fg_volumes), "fg_interpolation")

    nodes = create_slices(
        bg_volume,
        pos=pos,
        clim=bg_clim,
        cmap=bg_cmap,
        scale=scale,
        show_cbar=bool(show_cbar and cbar_type in ("bg", "base")),
        cbar_params=copy.deepcopy(cbar_params),
        interpolation=bg_interpolation,
        nancolor=nancolor,
        **kwargs,
    )

    for idx, volume in enumerate(fg_volumes):
        nodes = add_mask(
            nodes,
            volume,
            clim=fg_clims[idx],
            cmap=fg_cmaps[idx],
            interpolation=fg_interps[idx],
            alpha=alpha,
            excpt=excpt,
            preproc_func=preproc_func,
            show_cbar=bool(show_cbar and cbar_type in ("fg", "foreground") and idx == len(fg_volumes) - 1),
            cbar_params=copy.deepcopy(cbar_params),
            nancolor=nancolor,
        )
    return nodes


def create_surfaces(
    surfs,
    volume=None,
    value_type="depth",
    clim=None,
    cmap="jet",
    show_cbar=False,
    **kwargs,
):
    if not isinstance(surfs, list):
        surfs = [surfs]

    return [
        PlotlySurfacesSpec(
            surfs=surfs,
            volume=volume,
            value_type=value_type,
            clim=tuple(clim) if clim is not None else None,
            cmap=cmap,
            show_cbar=show_cbar,
            kwargs=copy.deepcopy(kwargs),
        )
    ]


def create_line_logs(logs, cmap="jet", line_width=8):
    """
    logs can be a np.ndarray (one log), or List of np.ndarray (muti-logs).
    each element's shape is (N, 3) or (N, 4).
    each row is (x, y, z) or (x, y, z, value)
    """
    if isinstance(logs, np.ndarray):
        logs = [logs]

    specs = []
    for log in logs:
        log = np.asarray(log)
        if log.shape[1] < 3:
            raise ValueError("each log must have at least three columns")
        specs.append(PlotlyLineLogSpec(log=log, cmap=cmap, line_width=line_width))
    return specs


def create_well_logs(*args, **kwargs):
    """
    use Mesh3D to create tube logs
    """
    raise NotImplementedError(
        "`create_well_logs` currently not supported in the jupyter, please run it with a .py file. If you must run in jupyter, please consider use `create_line_logs`"
    )  # noqa: E501


def create_points(points, color="red", size=3, sym="square"):
    points = np.asarray(points)
    return [PlotlyPointsSpec(points=points, color=color, size=size, sym=sym)]


def create_bodies(volume, level, margin: float = None, color="yellow"):
    if margin is not None and isinstance(volume, np.memmap):
        assert volume.mode != "r", "margin will modify the volume, set `mode='c'` instead of `mode='r'` in np.memmap"
    return [PlotlyBodySpec(volume=volume, level=level, margin=margin, color=color)]


def create_fault_skin(*args, **kwargs):
    raise NotImplementedError(
        "`create_fault_skin` currently not supported in the jupyter, please run it with a .py file."
    )  # noqa: E501


def plot3D(nodes, **kwargs):
    traces = compile_traces(nodes)

    size = kwargs.get("size", (900, 900))
    size = (size, size) if isinstance(size, (int, np.integer)) else size

    scene = kwargs.get("scene", {})
    scened = plotlyutils.make_3Dscene(**kwargs)
    for k, v in scened.items():
        scene.setdefault(k, v)

    fig = go.Figure(data=traces)

    fig.update_layout(
        height=size[0],
        width=size[1],
        scene=scene,
        margin=dict(l=5, r=5, t=5, b=5),
        showlegend=False,
    )

    savequality = kwargs.get("savequality", 1)
    show = kwargs.get("show", True)

    if show:
        fig.show(
            config={
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "custom_image",
                    "scale": savequality,
                }
            })

    return fig
