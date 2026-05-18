from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from cigvis.config import is_line_first
from . import utils as base_utils


_AXES = ('x', 'y', 'z')
_AXIS_ALIASES = {
    'x': 'x',
    'inline': 'x',
    'iline': 'x',
    'i': 'x',
    'y': 'y',
    'crossline': 'y',
    'xline': 'y',
    'j': 'y',
    'z': 'z',
    'time': 'z',
    'depth': 'z',
    'sample': 'z',
    't': 'z',
}
_DEFAULT_KEYS = {'default', '*', 'all'}
_SOURCE_DATA_KEYS = ('data', 'source', 'volume', 'array')
_SOURCE_AXES_KEYS = ('axes', 'dims', 'axis_order')


def normalize_axis_key(axis: str, *, allow_default: bool = False) -> str:
    key = str(axis).strip().lower()
    if allow_default and key in _DEFAULT_KEYS:
        return 'default'
    key = _AXIS_ALIASES.get(key, key)
    if key not in _AXES:
        raise ValueError("axis source keys must be one of 'x', 'y', 'z', or 'default'")
    return key


def is_axis_source_dict(value) -> bool:
    if not isinstance(value, dict):
        return False
    for key in value:
        try:
            normalize_axis_key(key, allow_default=True)
            return True
        except ValueError:
            continue
    return False


def is_source_spec(value) -> bool:
    """Return True for a single source spec such as {'data': arr, 'axes': ...}."""
    if not isinstance(value, dict):
        return False
    return any(key in value for key in _SOURCE_DATA_KEYS)


def _source_data(spec):
    if not is_source_spec(spec):
        return spec
    for key in _SOURCE_DATA_KEYS:
        if key in spec:
            return spec[key]
    return spec


def _source_axes(spec):
    if not is_source_spec(spec):
        return None
    for key in _SOURCE_AXES_KEYS:
        if key in spec:
            axes = spec[key]
            break
    else:
        return None
    axes = tuple(normalize_axis_key(axis) for axis in axes)
    if len(axes) != 3 or set(axes) != set(_AXES):
        raise ValueError(
            "source axes must contain the logical axes x/y/z exactly once, "
            "for example ('z', 'y', 'x')"
        )
    return axes


def normalize_display_range(display_range,
                            shape) -> Dict[str, Tuple[int, int]]:
    ranges = {
        'x': (0, int(shape[0])),
        'y': (0, int(shape[1])),
        'z': (0, int(shape[2])),
    }
    if display_range is None:
        return ranges
    if not isinstance(display_range, dict):
        raise TypeError("display_range must be a dict like {'z': (0, 900)}")

    for axis, value in display_range.items():
        axis = normalize_axis_key(axis)
        if value is None:
            continue
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("display_range values must be (start, stop)")
        start, stop = value
        start = 0 if start is None else int(start)
        stop = ranges[axis][1] if stop is None else int(stop)
        if start < 0 or stop > ranges[axis][1] or start >= stop:
            raise ValueError(
                f"display_range[{axis!r}] must satisfy "
                f"0 <= start < stop <= {ranges[axis][1]}, got ({start}, {stop})"
            )
        ranges[axis] = (start, stop)
    return ranges


def normalize_axis_sources(sources) -> Tuple[Dict[str, Any], Any]:
    if not is_axis_source_dict(sources):
        return {axis: sources for axis in _AXES}, sources

    axis_sources = {}
    default_source = None
    for key, value in sources.items():
        axis = normalize_axis_key(key, allow_default=True)
        if axis == 'default':
            default_source = value
        else:
            axis_sources[axis] = value

    if default_source is not None:
        for axis in _AXES:
            axis_sources.setdefault(axis, default_source)

    missing = [axis for axis in _AXES if axis not in axis_sources]
    if missing:
        raise ValueError(
            "axis source dict must provide x/y/z sources, or provide "
            f"a 'default' source. Missing: {', '.join(missing)}"
        )

    primary_source = default_source
    if primary_source is None:
        primary_source = next(axis_sources[axis] for axis in _AXES)
    return axis_sources, primary_source


def clim_source(source_or_sources):
    _sources, primary = normalize_axis_sources(source_or_sources)
    return _source_data(primary)


@dataclass
class _SourceMeta:
    source: Any
    shape: Tuple[int, int, int]
    rgb_type: int
    ndim: int
    channel_dim: Optional[int]
    axis_to_dim: Dict[str, int]
    custom_axes: bool = False


class SliceProvider:
    """
    Axis-aware 2D slice reader for numpy-like 3D/4D data sources.

    ``source`` can be a single volume or a dict such as
    ``{'x': iline_source, 'y': xline_source, 'z': time_source}``. A source may
    also declare its physical axis order, for example
    ``{'data': time_source, 'axes': ('z', 'y', 'x')}`` for data stored as
    ``(nt, nx, ni)``. Sources must expose numpy-like ``shape`` metadata and
    support slicing. ``ndim`` is used when available and inferred from
    ``shape`` otherwise. ``display_range`` is pushed into the source index so
    lazy/chunked sources only read the visible part of a slice.
    """

    def __init__(self,
                 source,
                 *,
                 preproc: Optional[Callable] = None,
                 forcefp32: bool = False,
                 display_range: Optional[Dict[str, Tuple[int, int]]] = None,
                 transpose_line_first: bool = True,
                 transpose_rgb: bool = True):
        self.line_first = is_line_first()
        self.preproc = preproc
        self.forcefp32 = bool(forcefp32)
        self.transpose_line_first = bool(transpose_line_first)
        self.transpose_rgb = bool(transpose_rgb)
        self.set_source(source, display_range=display_range, validate=False)

    def set_source(self,
                   source,
                   *,
                   display_range: Optional[Dict[str, Tuple[int, int]]] = None,
                   validate: bool = True):
        axis_sources, primary_source = normalize_axis_sources(source)
        raw_axis_sources = {
            axis: _source_data(src) for axis, src in axis_sources.items()
        }
        raw_primary_source = _source_data(primary_source)
        for src in set(map(id, raw_axis_sources.values())):
            # id-only loop avoids calling check_mmap repeatedly for the same object
            next_source = next(v for v in raw_axis_sources.values() if id(v) == src)
            base_utils.check_mmap(next_source)

        metas = {axis: self._make_meta(src) for axis, src in axis_sources.items()}
        primary_meta = self._make_meta(primary_source)
        shape = primary_meta.shape
        rgb_type = primary_meta.rgb_type
        for axis, meta in metas.items():
            if meta.shape != shape:
                raise ValueError(
                    f"axis source {axis!r} shape mismatch: {meta.shape} vs {shape}"
                )
            if meta.rgb_type != rgb_type:
                raise ValueError(
                    f"axis source {axis!r} RGB layout mismatch: "
                    f"{meta.rgb_type} vs {rgb_type}"
                )

        if validate and hasattr(self, 'shape') and shape != self.shape:
            raise ValueError(f"shape mismatch: {shape} vs {self.shape}")

        self.source = source
        self.axis_sources = raw_axis_sources
        self.primary_source = raw_primary_source
        self._metas = metas
        self.shape = shape
        self.rgb_type = rgb_type
        self.display_range = normalize_display_range(display_range, shape)

    def set_volume(self, new_volume, *, validate: bool = True, **_kwargs):
        self.set_source(new_volume,
                        display_range=self.display_range,
                        validate=validate)

    @property
    def clim_source(self):
        return self.primary_source

    def source_for_axis(self, axis: str):
        return self.axis_sources[normalize_axis_key(axis)]

    def get_shape2d(self, axis: str) -> Tuple[int, int]:
        axis = normalize_axis_key(axis)
        return tuple(self._axis_len(axis_key)
                     for axis_key in self._image_axes(axis))

    def _axis_len(self, axis: str) -> int:
        start, stop = self.display_range[axis]
        return stop - start

    def _make_meta(self, source) -> _SourceMeta:
        axes = _source_axes(source)
        source = _source_data(source)
        shape_attr = getattr(source, 'shape', None)
        if shape_attr is None:
            raise AttributeError("volume-like input must expose a shape attribute")
        ndim = getattr(source, 'ndim', len(shape_attr))
        assert ndim in (3, 4), f"vol.ndim must be 3 or 4, got {ndim}"
        if axes is None:
            shape, rgb_type = base_utils.get_shape(source, self.line_first)
        else:
            shape, rgb_type = self._shape_from_axes(source, axes)
        channel_dim = None
        if rgb_type == 1:
            channel_dim = 3
        elif rgb_type == 2:
            channel_dim = 0
        if axes is None:
            dim_x, dim_y, dim_z = (0, 1, 2) if self.line_first else (2, 1, 0)
            axis_to_dim = {'x': dim_x, 'y': dim_y, 'z': dim_z}
        else:
            axis_to_dim = {axis: i for i, axis in enumerate(axes)}
        return _SourceMeta(
            source=source,
            shape=tuple(shape),
            rgb_type=rgb_type,
            ndim=ndim,
            channel_dim=channel_dim,
            axis_to_dim=axis_to_dim,
            custom_axes=axes is not None,
        )

    def _shape_from_axes(self, source, axes) -> Tuple[Tuple[int, int, int], int]:
        shape = list(getattr(source, 'shape'))
        rgb_type = 0
        if len(shape) == 4:
            if shape[-1] in (3, 4):
                shape = shape[:3]
                rgb_type = 1
            elif shape[0] in (3, 4):
                shape = shape[1:]
                rgb_type = 2
            else:
                raise ValueError(f"Unknown RGB input type (shape={shape}).")
        if len(shape) != 3:
            raise ValueError(f"source axes expect a 3D volume shape, got {shape}")
        logical_shape = {
            axis: int(shape[i]) for i, axis in enumerate(axes)
        }
        return tuple(logical_shape[axis] for axis in _AXES), rgb_type

    def _get_slices(self, meta: _SourceMeta, axis: str, pos: int):
        slices = [slice(None)] * meta.ndim
        for axis_key in _AXES:
            dim = self._data_dim(meta, axis_key)
            if axis_key == axis:
                slices[dim] = pos
            else:
                start, stop = self.display_range[axis_key]
                slices[dim] = slice(start, stop)
        return tuple(slices)

    def _data_dim(self, meta: _SourceMeta, axis: str) -> int:
        dim = meta.axis_to_dim[axis]
        if meta.channel_dim is not None and dim >= meta.channel_dim:
            return dim + 1
        return dim

    def _wrap_preproc(self, img, meta: _SourceMeta):
        if self.transpose_rgb and not meta.custom_axes:
            if self.transpose_line_first and self.line_first and meta.rgb_type == 1:
                img = np.transpose(img, (1, 2, 0))
            elif (not self.line_first) and meta.rgb_type == 2:
                img = np.transpose(img, (1, 2, 0))

        if self.preproc is not None:
            img = self.preproc(img)

        if self.forcefp32:
            img = np.asarray(img)
            if img.dtype == np.float16:
                img = img.astype(np.float32)
        return img

    def _crop_display_range(self, axis: str, img, meta: _SourceMeta = None):
        expected = self.get_shape2d(axis)
        shape = getattr(img, 'shape', None)
        if shape is None or len(shape) < 2:
            return img

        spatial_dims = (0, 1)
        if (meta is not None and meta.rgb_type != 0 and
                not self.transpose_rgb and len(shape) >= 3 and shape[0] in (3, 4)):
            spatial_dims = (1, 2)

        current = tuple(shape[dim] for dim in spatial_dims)
        if current == expected:
            return img
        if any(shape[dim] < expected[i] for i, dim in enumerate(spatial_dims)):
            return img

        slices = [slice(None)] * len(shape)
        for i, dim in enumerate(spatial_dims):
            slices[dim] = slice(0, expected[i])
        return img[tuple(slices)]

    def _image_axes(self, axis: str):
        if self.transpose_line_first:
            if axis == 'x':
                return ('z', 'y')
            if axis == 'y':
                return ('z', 'x')
            return ('y', 'x')
        if self.line_first:
            if axis == 'x':
                return ('y', 'z')
            if axis == 'y':
                return ('x', 'z')
            return ('x', 'y')
        if axis == 'x':
            return ('z', 'y')
        if axis == 'y':
            return ('z', 'x')
        return ('y', 'x')

    def slice2d(self, axis: str, pos: int):
        axis = normalize_axis_key(axis)
        pos = int(np.round(pos))
        meta = self._metas[axis]
        img = meta.source[self._get_slices(meta, axis, pos)]
        if meta.custom_axes:
            img = self._orient_custom_slice(img, meta, axis)
        elif self.transpose_line_first and self.line_first:
            img = img.T
        img = self._wrap_preproc(img, meta)
        return self._crop_display_range(axis, img, meta)

    def _orient_custom_slice(self, img, meta: _SourceMeta, axis: str):
        raw_axes = tuple(
            axis_key for axis_key, _dim in sorted(
                (
                    (axis_key, meta.axis_to_dim[axis_key])
                    for axis_key in _AXES if axis_key != axis
                ),
                key=lambda item: item[1],
            )
        )
        desired_axes = self._image_axes(axis)
        if raw_axes == desired_axes and meta.channel_dim is None:
            return img

        if meta.channel_dim is None:
            perm = [raw_axes.index(axis_key) for axis_key in desired_axes]
        elif meta.channel_dim == 3:
            perm = [raw_axes.index(axis_key) for axis_key in desired_axes]
            perm.append(len(raw_axes))
        elif not self.transpose_rgb:
            perm = [0]
            perm.extend(1 + raw_axes.index(axis_key)
                        for axis_key in desired_axes)
        else:
            perm = [1 + raw_axes.index(axis_key) for axis_key in desired_axes]
            perm.append(0)
        return np.transpose(img, perm)

    def __call__(self, axis: str, pos: int, get_shape: bool = False):
        axis = normalize_axis_key(axis)
        if get_shape:
            return self.get_shape2d(axis)
        return self.slice2d(axis, pos)
