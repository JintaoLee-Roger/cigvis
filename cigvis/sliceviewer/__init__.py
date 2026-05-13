# Copyright (c) 2025 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
"""SSH-friendly interactive 2D slice viewer based on Panel + Plotly.

The viewer maps a 2D/3D/4D array to a browser-based 2D image. Users can
choose which two dimensions are displayed and which remaining dimensions are
fixed by index.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .nodes import MaskSpec, SliceNode

try:
    import plotly.graph_objects as go
    from cigvis import colormap as _cmap_mod
    from .viewer import build_layout, create_server, link, show
except ImportError as e:
    raise ImportError(
        "SliceViewer requires plotly and panel.\n"
        "Install: pip install \"cigvis[sliceviewer]\""
    ) from e

__all__ = [
    "create_slice",
    "add_mask",
    "add_horizon",
    "add_fault",
    "add_well",
    "add_scatter",
    "link",
    "show",
    "build_layout",
    "create_server",
]


def create_slice(
    volume: np.ndarray,
    display_axes: Optional[Sequence[int]] = None,
    indices: Optional[Union[Dict[int, int], Sequence[int]]] = None,
    cmap: str = "gray",
    clim: Optional[List] = None,
    aspect: Union[str, float] = 1.0,
    axis_labels: Optional[Sequence[str]] = None,
) -> List[SliceNode]:
    """
    Create a 2D slice node from a 2D/3D/4D array.

    Parameters
    ----------
    volume : np.ndarray
        2D, 3D, or 4D array.
    display_axes : Sequence[int], optional
        Two dimensions to display as ``(y_axis, x_axis)``. If omitted, the
        two largest dimensions are displayed.
    indices : dict or sequence, optional
        Initial fixed indices for dimensions not shown in ``display_axes``.
        A dict maps data-axis index to fixed index, e.g. ``{0: 1}``; a
        sequence is read in hidden-axis order.
    cmap : str
        Colormap name.
    clim : List, optional
        [vmin, vmax]. Default: auto from p2/p98 of the initial 2D frame.
    aspect : {"auto"} or float
        Plot aspect. ``1.0`` keeps equal sample spacing; ``"auto"`` lets the
        browser fill the available plotting area.
    axis_labels : Sequence[str], optional
        Human-readable labels for data axes. Defaults to ``dim 0``,
        ``dim 1``, ... .
    """
    volume = np.asarray(volume)
    if volume.ndim < 2 or volume.ndim > 4:
        raise ValueError("volume must be a 2D, 3D, or 4D array")
    if min(volume.shape) <= 0:
        raise ValueError(f"volume shape must be non-empty, got {volume.shape}")

    display_axes = _normalize_display_axes(display_axes, volume.shape)
    node_indices = _normalize_indices(indices, display_axes, volume.shape)
    node = SliceNode(
        volume=volume,
        display_axes=display_axes,
        indices=node_indices,
        cmap=_cmap_mod.get_cmap_from_str(cmap),
        clim=(0.0, 1.0),
        aspect=_normalize_aspect(aspect),
        axis_labels=_normalize_axis_labels(axis_labels, volume.ndim),
    )
    node.clim = _finite_range(node.get_frame(), percentiles=(2, 98)) \
        if clim is None else _normalize_clim(clim)
    return [node]


def add_mask(
    nodes: List,
    volume: np.ndarray,
    cmap=None,
    clim=None,
    alpha: Optional[float] = 0.7,
    excpt: Optional[str] = "min",
) -> List:
    """Add one overlay volume on top of the slice."""
    if cmap is None:
        raise ValueError("'cmap' cannot be None")

    slice_nodes = [nd for nd in nodes if isinstance(nd, SliceNode)]
    if not slice_nodes:
        raise ValueError("nodes must contain a SliceNode (from create_slice)")
    sn = slice_nodes[0]

    volume = np.asarray(volume)
    if volume.shape != sn.volume.shape:
        raise ValueError(
            f"Mask shape {volume.shape} must match volume shape {sn.volume.shape}"
        )

    clim = _finite_range(volume) if clim is None else _normalize_clim(clim)
    cmap_obj = _cmap_mod.get_cmap_from_str(cmap) if alpha is None \
        else _cmap_mod.fast_set_cmap(cmap, alpha, excpt)
    sn.masks.append(MaskSpec(volume=volume, cmap=cmap_obj, clim=tuple(clim)))

    return nodes


def add_horizon(
    x,
    y,
    name: str = "horizon",
    color: str = "yellow",
    width: float = 1.5,
) -> List[go.Scatter]:
    """Add a horizon line annotation."""
    return [
        go.Scatter(
            x=np.asarray(x),
            y=np.asarray(y),
            mode="lines",
            name=name,
            line=dict(color=color, width=width),
        )
    ]


def add_fault(
    x,
    y,
    name: str = "fault",
    color: str = "red",
    width: float = 1.5,
) -> List[go.Scatter]:
    """Add a fault line annotation."""
    return [
        go.Scatter(
            x=np.asarray(x),
            y=np.asarray(y),
            mode="lines",
            name=name,
            line=dict(color=color, width=width),
        )
    ]


def add_well(
    x,
    y,
    name: str = "well",
    color: str = "white",
    size: float = 6,
) -> List[go.Scatter]:
    """Add well positions as scatter points."""
    x = np.asarray(x)
    return [
        go.Scatter(
            x=x,
            y=np.asarray(y),
            mode="markers+text",
            name=name,
            text=[name] * len(x),
            textposition="top center",
            marker=dict(color=color, size=size),
        )
    ]


def add_scatter(
    x,
    y,
    name: str = "scatter",
    mode: str = "markers",
    color: str = "cyan",
    size: float = 6,
    **kwargs,
) -> List[go.Scatter]:
    """Add a generic scatter or line annotation."""
    marker = dict(color=color, size=size)
    marker.update(kwargs.pop("marker", {}))
    return [
        go.Scatter(
            x=np.asarray(x),
            y=np.asarray(y),
            mode=mode,
            name=name,
            marker=marker,
            **kwargs,
        )
    ]


def _normalize_display_axes(
    display_axes: Optional[Sequence[int]],
    shape: Tuple[int, ...],
) -> Tuple[int, int]:
    ndim = len(shape)
    if display_axes is not None:
        axes = tuple(_normalize_axis(ax, ndim) for ax in display_axes)
        if len(axes) != 2 or axes[0] == axes[1]:
            raise ValueError("display_axes must be two different axes")
        return axes

    if ndim == 2:
        return (0, 1)

    axes = sorted(range(ndim), key=lambda ax: shape[ax], reverse=True)[:2]
    return tuple(sorted(axes))


def _normalize_indices(
    indices: Optional[Union[Dict[int, int], Sequence[int]]],
    display_axes: Tuple[int, int],
    shape: Tuple[int, ...],
) -> Dict[int, int]:
    ndim = len(shape)
    hidden_axes = [axis for axis in range(ndim) if axis not in display_axes]
    out = {axis: shape[axis] // 2 for axis in hidden_axes}
    if indices is None:
        return out

    if isinstance(indices, dict):
        for axis, idx in indices.items():
            ax = _normalize_axis(axis, ndim)
            if ax not in out:
                raise ValueError(
                    f"indices contains displayed axis dim {ax}; only hidden "
                    "axes need fixed indices"
                )
            out[ax] = _clamp_index(idx, shape[ax])
        return out

    values = list(indices)
    if len(values) == len(hidden_axes):
        for ax, idx in zip(hidden_axes, values):
            out[ax] = _clamp_index(idx, shape[ax])
    else:
        raise ValueError(
            "indices sequence must have one value per hidden axis"
        )
    return out


def _normalize_axis(axis: int, ndim: int) -> int:
    try:
        axis = int(axis)
    except (TypeError, ValueError) as exc:
        raise ValueError("axis must be an integer data-axis index") from exc

    if axis < 0 or axis >= ndim:
        raise ValueError(f"axis {axis} is invalid for ndim={ndim}")
    return axis


def _clamp_index(idx: int, size: int) -> int:
    return int(np.clip(int(idx), 0, size - 1))


def _normalize_aspect(aspect: Union[str, float]) -> Union[str, float]:
    if isinstance(aspect, str):
        aspect = aspect.strip().lower()
        if aspect in ("auto", "free"):
            return "auto"
        if aspect == "equal":
            return 1.0
        raise ValueError("aspect must be 'auto', 'equal', or a positive number")
    aspect = float(aspect)
    if not np.isfinite(aspect) or aspect <= 0:
        raise ValueError("aspect must be positive")
    return aspect


def _normalize_axis_labels(axis_labels: Optional[Sequence[str]],
                           ndim: int) -> Tuple[str, ...]:
    if axis_labels is None:
        return tuple(f"dim {axis}" for axis in range(ndim))
    labels = tuple(str(label) for label in axis_labels)
    if len(labels) != ndim:
        raise ValueError(f"axis_labels must contain {ndim} labels")
    if any(not label.strip() for label in labels):
        raise ValueError("axis_labels cannot contain empty labels")
    return labels


def _finite_range(
    data: np.ndarray,
    percentiles: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    finite = np.asarray(data)[np.isfinite(data)]
    if finite.size == 0:
        return (0.0, 1.0)
    if percentiles is None:
        clim = (float(finite.min()), float(finite.max()))
    else:
        clim = (
            float(np.percentile(finite, percentiles[0])),
            float(np.percentile(finite, percentiles[1])),
        )
    return _normalize_clim(clim)


def _normalize_clim(clim: Sequence) -> Tuple[float, float]:
    arr = np.asarray(clim, dtype=float)
    if arr.shape != (2,):
        raise ValueError("clim must be a two-value sequence: [vmin, vmax]")
    vmin, vmax = float(arr[0]), float(arr[1])
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return (0.0, 1.0)
    if vmin > vmax:
        vmin, vmax = vmax, vmin
    if vmin == vmax:
        pad = abs(vmin) * 0.01 or 1.0
        vmin -= pad
        vmax += pad
    return (vmin, vmax)
