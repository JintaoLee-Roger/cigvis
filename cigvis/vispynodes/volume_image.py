from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from cigvis import colormap
from cigvis.utils import utils
from cigvis.utils.slice_provider import SliceProvider
from .axis_aligned_image import AxisAlignedImage, InteractiveLine


_AXES = ('x', 'y', 'z')


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


def _normalize_axes(axes=None):
    if axes is None:
        return _AXES
    if isinstance(axes, str):
        axes = [axes]
    out = []
    for axis in axes:
        axis = str(axis).lower()
        if axis not in _AXES:
            raise ValueError("axis must be one of 'x', 'y', or 'z'")
        if axis not in out:
            out.append(axis)
    return tuple(out)


# -------------------------
# Overlay spec
# -------------------------
@dataclass
class OverlaySpec:
    name: str

    # for volume3d:
    volume: Optional[np.ndarray] = None
    preproc: Optional[Callable] = None
    forcefp32: bool = True  # masks often use fp16->fp32 path

    cmap: Any = 'grays'
    cmap_name: Any = None
    axis_cmaps: Optional[Dict[str, Any]] = None
    axis_cmap_names: Optional[Dict[str, Any]] = None
    axes: Tuple[str, ...] = _AXES
    clim: Optional[Tuple[float, float]] = None
    interpolation: str = 'nearest'
    method: str = 'auto'
    texture_format: Optional[str] = 'auto'

    dtype: Any = np.float32

    def cmap_for_axis(self, axis: str):
        if self.axis_cmaps is not None and axis in self.axis_cmaps:
            return self.axis_cmaps[axis]
        return self.cmap

    def cmap_name_for_axis(self, axis: str):
        if self.axis_cmap_names is not None and axis in self.axis_cmap_names:
            return self.axis_cmap_names[axis]
        return self.cmap_name


# -------------------------
# VolumeImage
# -------------------------
class VolumeImage:
    """
    Manager for one base volume and multiple overlays, and for creating
    axis-aligned slice nodes (AxisAlignedImage) while keeping them independent.

    Key improvement:
      - Use SliceProvider (mutable volume reference) so replacing overlay volumes
        does NOT require swapping image_funcs closures.
    """

    def __init__(
        self,
        volume: np.ndarray,
        *,
        preproc: Optional[Callable] = None,
        cmap: Any = 'grays',
        clim: Optional[Union[List, Tuple]] = None,
        interpolation: str = 'linear',
        method: str = 'auto',
        texture_format: Optional[str] = None,
        display_range: Optional[Dict[str, Tuple[int, int]]] = None,
    ):
        self.volume = volume

        self.base_preproc = preproc
        self.base_cmap_name = cmap if isinstance(cmap, str) else getattr(cmap, 'name', None)
        self.base_cmap = colormap.cmap_to_vispy(cmap)
        self.base_clim = clim
        self.base_interpolation = interpolation
        self.method = method
        self.texture_format = texture_format

        # Providers: base + each volume3d overlay has its own provider
        self._providers: Dict[str, SliceProvider] = {}
        self._providers['__base__'] = SliceProvider(
            self.volume,
            preproc=self.base_preproc,
            forcefp32=False,
            display_range=display_range,
        )
        self.shape = self._providers['__base__'].shape
        self.display_range = self._providers['__base__'].display_range

        # name -> OverlaySpec
        self._overlays: Dict[str, OverlaySpec] = {}

        # created nodes: {'x': [AxisAlignedImage...], 'y':..., 'z':...}
        self._slices: Dict[str, List[AxisAlignedImage]] = {'x': [], 'y': [], 'z': []}

        # mapping (axis, node_index) -> overlay_name -> overlay_image_index_in_node
        self._overlay_indices: Dict[Tuple[str, int], Dict[str, int]] = {}

    # -------------------------
    # overlay registration
    # -------------------------
    def add_overlay_volume(
        self,
        *,
        name: str,
        volume: np.ndarray,
        cmap: Any,
        clim: Optional[Union[List, Tuple]] = None,
        interpolation: str = 'nearest',
        preproc: Optional[Callable] = None,
        method: Optional[str] = None,
        texture_format: Optional[str] = None,
        forcefp32: bool = True,
        axes=None,
        cmap_names: Any = None,
    ):
        provider = SliceProvider(
            volume,
            preproc=preproc,
            forcefp32=forcefp32,
            display_range=self.display_range,
        )
        if provider.shape != self.shape:
            raise ValueError(
                f"Overlay volume '{name}' shape mismatch: {provider.shape} vs {self.shape}"
            )

        if isinstance(cmap, dict):
            cmap_by_axis = {str(axis).lower(): value for axis, value in cmap.items()}
            axes = _normalize_axes(cmap_by_axis.keys() if axes is None else axes)
            axis_cmaps = {}
            axis_cmap_names = {}
            for axis in axes:
                if axis not in cmap_by_axis:
                    continue
                axis_cmaps[axis] = colormap.cmap_to_vispy(cmap_by_axis[axis])
                if isinstance(cmap_names, dict) and axis in cmap_names:
                    axis_cmap_names[axis] = cmap_names[axis]
                else:
                    cmap_value = cmap_by_axis[axis]
                    axis_cmap_names[axis] = cmap_value if isinstance(cmap_value, str) else getattr(cmap_value, 'name', None)
            if not axis_cmaps:
                raise ValueError("cmap dict must contain at least one of 'x', 'y', or 'z'")
            first_axis = next(iter(axis_cmaps))
            cmap_v = axis_cmaps[first_axis]
            cmap_name = axis_cmap_names.get(first_axis)
        else:
            axes = _normalize_axes(axes)
            axis_cmaps = None
            axis_cmap_names = None
            cmap_v = colormap.cmap_to_vispy(cmap)
            cmap_name = cmap_names if cmap_names is not None else (cmap if isinstance(cmap, str) else getattr(cmap, 'name', None))

        spec = OverlaySpec(
            name=name,
            volume=volume,
            preproc=preproc,
            forcefp32=forcefp32,
            cmap=cmap_v,
            cmap_name=cmap_name,
            axis_cmaps=axis_cmaps,
            axis_cmap_names=axis_cmap_names,
            axes=tuple(axes),
            clim=tuple(clim) if clim is not None else None,
            interpolation=interpolation,
            method=method or self.method,
            texture_format=texture_format if texture_format is not None else (self.texture_format or 'auto'),
        )
        self._overlays[name] = spec

        # create provider (mutable ref)
        self._providers[name] = provider

        # If slices already exist, attach this overlay to existing nodes.
        if any(len(v) for v in self._slices.values()):
            self._attach_overlay_to_existing_nodes(name)


    # -------------------------
    # slice creation
    # -------------------------
    def create_slices(
        self,
        x_pos: Optional[Union[List, int, float]] = None,
        y_pos: Optional[Union[List, int, float]] = None,
        z_pos: Optional[Union[List, int, float]] = None,
        pos: Optional[Union[Dict[str, List[int]], List, Tuple]] = None,
    ) -> Dict[str, List[AxisAlignedImage]]:
        if pos is not None:
            pos = self._normalize_pos(pos)
            x_pos = pos.get('x', x_pos)
            y_pos = pos.get('y', y_pos)
            z_pos = pos.get('z', z_pos)

        if x_pos is None and y_pos is None and z_pos is None:
            x_pos = [self.display_range['x'][0]]
            y_pos = [self.display_range['y'][0]]
            z_pos = [self.display_range['z'][1] - 1]

        axis_slices = {'x': x_pos, 'y': y_pos, 'z': z_pos}

        def _limit(axis: str):
            start, stop = self.display_range[axis]
            return (start, stop - 1)

        # clear old
        self._slices = {'x': [], 'y': [], 'z': []}
        self._overlay_indices = {}

        for axis, pos_list in axis_slices.items():
            if pos_list is None:
                continue
            if isinstance(pos_list, (int, float)):
                pos_list = [pos_list]
            for p in pos_list:
                p = int(np.round(p))
                node = self._build_axis_node(axis=axis, pos=p, limit=_limit(axis))
                self._slices[axis].append(node)

        # attach all overlays
        for name in list(self._overlays.keys()):
            self._attach_overlay_to_existing_nodes(name)


        return self._slices

    def _normalize_pos(self, pos):
        if isinstance(pos, tuple):
            pos = list(pos)
        if isinstance(pos, list):
            assert len(pos) == 3
            if isinstance(pos[0], (list, tuple)):
                x, y, z = pos
            else:
                x, y, z = [pos[0]], [pos[1]], [pos[2]]
            return {'x': x, 'y': y, 'z': z}
        if isinstance(pos, dict):
            return {
                'x': pos.get('x'),
                'y': pos.get('y'),
                'z': pos.get('z'),
            }
        raise TypeError("pos must be a list/tuple or dict")

    def nodes(
        self,
        *,
        intersection_lines: bool = False,
        line_color=(1, 1, 1),
        line_width: float = 2.0,
    ):
        out: List[Any] = []
        out += self._slices['x']
        out += self._slices['y']
        out += self._slices['z']
        if intersection_lines:
            out += self._add_intersection_line(
                [self._slices['x'], self._slices['y'], self._slices['z']],
                line_color,
                line_width,
            )
        return out

    # -------------------------
    # workflow APIs
    # -------------------------
    def replace_overlay_volume(
        self,
        name: str,
        new_volume: np.ndarray,
        *,
        preproc: Optional[Callable] = None,
        refresh: bool = True,
        validate: bool = True,
        reinit_meta_if_needed: bool = False,
    ):
        """
        Workflow-friendly: vol -> AI model -> new mask volume (NEW reference) -> update.

        With SliceProvider, we only swap provider.volume; we DO NOT replace image_funcs.
        """
        if name not in self._overlays:
            raise KeyError(f"Overlay '{name}' not registered.")
        spec = self._overlays[name]

        new_provider = SliceProvider(
            new_volume,
            preproc=preproc or spec.preproc,
            forcefp32=spec.forcefp32,
            display_range=self.display_range,
        )
        if validate and new_provider.shape != self.shape:
            raise ValueError(
                f"Overlay '{name}' shape mismatch: {new_provider.shape} vs base {self.shape}"
            )

        # update spec reference
        spec.volume = new_volume
        if preproc is not None:
            spec.preproc = preproc

        # update provider reference
        if name not in self._providers:
            # should not happen, but be robust
            self._providers[name] = new_provider
        else:
            prov = self._providers[name]
            if preproc is not None:
                prov.preproc = preproc
            prov.set_volume(new_volume, validate=validate)

        if refresh:
            self.refresh_overlay(name)

    def refresh_overlay(self, name: str):
        """
        Re-upload current slice for a 'volume3d' overlay.
        Useful for in-place edits, or when you want immediate refresh without dragging.
        """
        if name not in self._overlays:
            raise KeyError(f"Overlay '{name}' not registered.")
        spec = self._overlays[name]

        for axis in ('x', 'y', 'z'):
            for i, node in enumerate(self._slices[axis]):
                overlay_idx = self._overlay_indices.get((axis, i), {}).get(name, None)
                if overlay_idx is None:
                    continue
                # node.image_funcs[overlay_idx] is bound to provider, so it slices latest volume
                image = node.overlaid_images[overlay_idx]
                image.cmap = spec.cmap_for_axis(axis)
                if spec.clim is not None:
                    image.clim = list(spec.clim)
                image.interpolation = spec.interpolation
                image.set_data(node.image_funcs[overlay_idx](node.pos))
                _set_visual_metadata(
                    image,
                    _cigvis_cmap_name=spec.cmap_name_for_axis(axis),
                    _cigvis_interpolation=spec.interpolation,
                )

    def remove_overlay(self, name: str) -> bool:
        """Remove an overlay volume from all existing slice nodes."""
        if name not in self._overlays:
            return False

        for axis in ('x', 'y', 'z'):
            for i, node in enumerate(self._slices[axis]):
                key = (axis, i)
                mapping = self._overlay_indices.get(key)
                if not mapping or name not in mapping:
                    continue
                overlay_idx = mapping.pop(name)
                node.remove_mask(overlay_idx)
                for other, idx in list(mapping.items()):
                    if idx > overlay_idx:
                        mapping[other] = idx - 1
                if not mapping:
                    self._overlay_indices.pop(key, None)

        self._overlays.pop(name, None)
        self._providers.pop(name, None)
        return True

    # -------------------------
    # internal helpers
    # -------------------------
    def _resolve_clim(self,
                      vol: np.ndarray,
                      clim: Optional[Union[List, Tuple]],
                      provider: SliceProvider = None):
        if clim is None or clim == 'auto':
            source = provider.clim_source if provider is not None else vol
            return utils.auto_clim(source)
        return tuple(clim)

    def _bind_provider_func(self, provider_name: str, axis: str):
        """
        Return a callable with signature (pos, get_shape=False) -> slice2d or shape2d,
        without capturing ndarray reference. It captures provider object only.
        """
        prov = self._providers[provider_name]
        ax = axis.lower()

        def _f(pos, get_shape: bool = False, _prov=prov, _ax=ax):
            return _prov(_ax, pos, get_shape=get_shape)

        return _f

    def _build_axis_node(self, *, axis: str, pos: int, limit: Tuple[int, int]) -> AxisAlignedImage:
        # base is always provider '__base__'
        image_funcs = [self._bind_provider_func('__base__', axis)]

        cmaps = [self.base_cmap]
        clims = [
            self._resolve_clim(
                self.volume,
                self.base_clim,
                self._providers['__base__'],
            )
        ]
        interps = [self.base_interpolation]

        node = AxisAlignedImage(
            image_funcs=image_funcs,
            axis=axis,
            pos=pos,
            limit=limit,
            cmaps=cmaps,
            clims=clims,
            interpolation=interps,
            method=self.method,
            texture_format=self.texture_format,
            display_range=self.display_range,
        )
        _set_visual_metadata(
            node,
            _cigvis_volume_image=self,
            _cigvis_display_range=self.display_range,
            _cigvis_cmap_name=self.base_cmap_name,
            _cigvis_interpolation=self.base_interpolation,
        )
        for image in getattr(node, 'overlaid_images', [node]):
            _set_visual_metadata(
                image,
                _cigvis_cmap_name=self.base_cmap_name,
                _cigvis_interpolation=self.base_interpolation,
            )
        return node

    def _attach_overlay_to_existing_nodes(self, name: str):
        spec = self._overlays[name]
        for axis in ('x', 'y', 'z'):
            if axis not in spec.axes:
                continue
            for i, node in enumerate(self._slices[axis]):
                if (axis, i) not in self._overlay_indices:
                    self._overlay_indices[(axis, i)] = {}
                if name in self._overlay_indices[(axis, i)]:
                    continue

                # Append provider-bound image_func + a child Image
                node.unfreeze()
                node.image_funcs.append(self._bind_provider_func(name, axis))
                node.freeze()

                # create child overlay image visual
                from vispy.scene.visuals import Image as VispyImage

                overlay = VispyImage(
                    parent=node,
                    cmap=spec.cmap_for_axis(axis),
                    clim=list(spec.clim) if spec.clim is not None else list(
                        self._resolve_clim(spec.volume, None,
                                           self._providers[name])),
                    interpolation=spec.interpolation,
                    method=spec.method,
                    texture_format=spec.texture_format,
                )
                node.overlaid_images.append(overlay)

                overlay_idx = len(node.overlaid_images) - 1
                self._overlay_indices[(axis, i)][name] = overlay_idx

                # initialize overlay data to current slice
                overlay.set_data(node.image_funcs[overlay_idx](node.pos))
                _set_visual_metadata(
                    overlay,
                    _cigvis_cmap_name=spec.cmap_name_for_axis(axis),
                    _cigvis_interpolation=spec.interpolation,
                )


    def _add_intersection_line(self, image_nodes, line_color=(1, 1, 1), line_width: float = 2.0):
        lines_nodes = []

        # X-Y intersection lines
        for x_img, y_img in product(image_nodes[0], image_nodes[1]):
            line = InteractiveLine(
                ('x', 'y'),
                self.shape,
                color=line_color,
                width=line_width,
                antialias=True,
                display_range=self.display_range,
            )
            line.link_image(x_img)
            line.link_image(y_img)
            line.refresh()
            lines_nodes.append(line)

        # X-Z intersection lines
        for x_img, z_img in product(image_nodes[0], image_nodes[2]):
            line = InteractiveLine(
                ('x', 'z'),
                self.shape,
                color=line_color,
                width=line_width,
                antialias=True,
                display_range=self.display_range,
            )
            line.link_image(x_img)
            line.link_image(z_img)
            line.refresh()
            lines_nodes.append(line)

        # Y-Z intersection lines
        for y_img, z_img in product(image_nodes[1], image_nodes[2]):
            line = InteractiveLine(
                ('y', 'z'),
                self.shape,
                color=line_color,
                width=line_width,
                antialias=True,
                display_range=self.display_range,
            )
            line.link_image(y_img)
            line.link_image(z_img)
            line.refresh()
            lines_nodes.append(line)

        # contour lines for X Images
        for x_img in image_nodes[0]:
            line = InteractiveLine(
                ('x',),
                self.shape,
                color=line_color,
                width=line_width,
                antialias=True,
                display_range=self.display_range,
            )
            line.link_image(x_img)
            line.refresh()
            lines_nodes.append(line)

        # contour lines for Y Images
        for y_img in image_nodes[1]:
            line = InteractiveLine(
                ('y',),
                self.shape,
                color=line_color,
                width=line_width,
                antialias=True,
                display_range=self.display_range,
            )
            line.link_image(y_img)
            line.refresh()
            lines_nodes.append(line)

        # contour lines for Z Images
        for z_img in image_nodes[2]:
            line = InteractiveLine(
                ('z',),
                self.shape,
                color=line_color,
                width=line_width,
                antialias=True,
                display_range=self.display_range,
            )
            line.link_image(z_img)
            line.refresh()
            lines_nodes.append(line)

        return lines_nodes
