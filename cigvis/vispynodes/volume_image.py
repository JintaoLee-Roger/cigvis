from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from cigvis import is_line_first
from cigvis import colormap
from cigvis.utils import utils
from .axis_aligned_image import AxisAlignedImage, InteractiveLine


# -------------------------
# SliceProvider: no closure captures ndarray
# -------------------------
@dataclass
class SliceProvider:
    """
    Provide 2D slices from a 3D/4D volume with cached metadata (line_first, rgb_type,
    axis mapping, etc.). The key idea: the volume reference is mutable via set_volume(),
    so you can swap ndarray references without replacing image_funcs.
    """
    volume: np.ndarray
    preproc: Optional[Callable] = None
    forcefp32: bool = False

    def __post_init__(self):
        self._init_meta()

    def _init_meta(self):
        self.line_first = is_line_first()
        assert self.volume.ndim in (3, 4), f"vol.ndim must be 3 or 4, got {self.volume.ndim}"
        self.ndim = self.volume.ndim

        # NOTE: depending on cigvis.utils implementation, you might need:
        # self.shape, self.rgb_type = utils.utils.get_shape(self.volume, self.line_first)
        self.shape, self.rgb_type = utils.get_shape(self.volume, self.line_first)

        # rgb_type:
        # 0 for (n1, n2, n3)
        # 1 for (n1, n2, n3, 3/4)
        # 2 for (3/4, n1, n2, n3)
        self.channel_dim = None
        if self.rgb_type == 1:
            self.channel_dim = 3
        elif self.rgb_type == 2:
            self.channel_dim = 0

        dim_x, dim_y, dim_z = (0, 1, 2) if self.line_first else (2, 1, 0)
        self.axis_to_dim = {'x': dim_x, 'y': dim_y, 'z': dim_z}

    def set_volume(self, new_volume: np.ndarray, *, validate: bool = True, reinit_meta_if_needed: bool = False):
        """
        Swap the backing ndarray reference. For typical workflow (same shape/dtype layout),
        validate=True is enough. If you might change ndim/rgb layout, set reinit_meta_if_needed=True.
        """
        if validate:
            if new_volume.shape != self.volume.shape:
                raise ValueError(f"shape mismatch: {new_volume.shape} vs {self.volume.shape}")
            if new_volume.ndim != self.volume.ndim:
                raise ValueError(f"ndim mismatch: {new_volume.ndim} vs {self.volume.ndim}")

        self.volume = new_volume

        if reinit_meta_if_needed:
            # only needed when ndim/rgb layout may change; otherwise keep cached meta for speed
            self._init_meta()

    def get_shape2d(self, axis: str) -> Tuple[int, int]:
        axis = axis.lower()
        if axis == 'x':
            return (self.shape[1], self.shape[2])
        if axis == 'y':
            return (self.shape[0], self.shape[2])
        if axis == 'z':
            return (self.shape[0], self.shape[1])
        raise ValueError("axis must be x/y/z")

    def _wrap_preproc(self, x: np.ndarray) -> np.ndarray:
        # Keep same RGB transpose behavior as original get_image_func
        if self.line_first and self.rgb_type == 1:
            x = np.transpose(x, (1, 2, 0))
        elif (not self.line_first) and self.rgb_type == 2:
            x = np.transpose(x, (1, 2, 0))

        if self.preproc is not None:
            x = self.preproc(x)

        if self.forcefp32:
            x = np.asarray(x)
            if x.dtype == np.float16:
                x = x.astype(np.float32)
        return x

    def _get_slices(self, axis: str, pos: int):
        dim = self.axis_to_dim[axis]
        slices = [slice(None)] * self.ndim
        if self.channel_dim is not None and dim >= self.channel_dim:
            slices[dim + 1] = pos
        else:
            slices[dim] = pos
        return tuple(slices)

    def slice2d(self, axis: str, pos: int) -> np.ndarray:
        axis = axis.lower()
        pos = int(np.round(pos))
        s = self._get_slices(axis, pos)

        # Keep old behavior: when line_first, take transpose after slicing
        if self.line_first:
            return self._wrap_preproc(self.volume[s].T)
        else:
            return self._wrap_preproc(self.volume[s])

    def __call__(self, axis: str, pos: int, get_shape: bool = False):
        if get_shape:
            return self.get_shape2d(axis)
        return self.slice2d(axis, pos)


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
    clim: Optional[Tuple[float, float]] = None
    interpolation: str = 'nearest'
    method: str = 'auto'
    texture_format: Optional[str] = 'auto'

    dtype: Any = np.float32


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
    ):
        utils.check_mmap(volume)
        self.volume = volume

        self.base_preproc = preproc
        self.base_cmap = colormap.cmap_to_vispy(cmap)
        self.base_clim = clim
        self.base_interpolation = interpolation
        self.method = method
        self.texture_format = texture_format

        lf = is_line_first()
        # NOTE: depending on cigvis.utils, may need utils.utils.get_shape
        self.shape, _ = utils.get_shape(volume, lf)

        # Providers: base + each volume3d overlay has its own provider
        self._providers: Dict[str, SliceProvider] = {}
        self._providers['__base__'] = SliceProvider(self.volume, preproc=self.base_preproc, forcefp32=False)

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
    ):
        utils.check_mmap(volume)
        if volume.shape != self.volume.shape:
            raise ValueError(f"Overlay volume '{name}' shape mismatch: {volume.shape} vs {self.volume.shape}")

        cmap_v = colormap.cmap_to_vispy(cmap)
        spec = OverlaySpec(
            name=name,
            volume=volume,
            preproc=preproc,
            forcefp32=forcefp32,
            cmap=cmap_v,
            clim=tuple(clim) if clim is not None else None,
            interpolation=interpolation,
            method=method or self.method,
            texture_format=texture_format if texture_format is not None else (self.texture_format or 'auto'),
        )
        self._overlays[name] = spec

        # create provider (mutable ref)
        self._providers[name] = SliceProvider(volume, preproc=preproc, forcefp32=forcefp32)

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
        pos: Optional[Dict[str, List[int]]] = None,
    ) -> Dict[str, List[AxisAlignedImage]]:
        if pos is not None:
            x_pos = pos.get('x', x_pos)
            y_pos = pos.get('y', y_pos)
            z_pos = pos.get('z', z_pos)

        if x_pos is None and y_pos is None and z_pos is None:
            x_pos = [0]
            y_pos = [0]
            z_pos = [self.shape[-1] - 1]

        axis_slices = {'x': x_pos, 'y': y_pos, 'z': z_pos}

        def _limit(axis: str):
            if axis == 'x':
                return (0, self.shape[0] - 1)
            if axis == 'y':
                return (0, self.shape[1] - 1)
            return (0, self.shape[2] - 1)

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

        utils.check_mmap(new_volume)
        if validate and new_volume.shape != self.volume.shape:
            raise ValueError(f"Overlay '{name}' shape mismatch: {new_volume.shape} vs base {self.volume.shape}")

        # update spec reference
        spec.volume = new_volume
        if preproc is not None:
            spec.preproc = preproc

        # update provider reference
        if name not in self._providers:
            # should not happen, but be robust
            self._providers[name] = SliceProvider(new_volume, preproc=spec.preproc, forcefp32=spec.forcefp32)
        else:
            prov = self._providers[name]
            if preproc is not None:
                prov.preproc = preproc
            prov.set_volume(new_volume, validate=validate, reinit_meta_if_needed=reinit_meta_if_needed)

        if refresh:
            self.refresh_overlay(name)

    def refresh_overlay(self, name: str):
        """
        Re-upload current slice for a 'volume3d' overlay.
        Useful for in-place edits, or when you want immediate refresh without dragging.
        """
        if name not in self._overlays:
            raise KeyError(f"Overlay '{name}' not registered.")

        for axis in ('x', 'y', 'z'):
            for i, node in enumerate(self._slices[axis]):
                overlay_idx = self._overlay_indices.get((axis, i), {}).get(name, None)
                if overlay_idx is None:
                    continue
                # node.image_funcs[overlay_idx] is bound to provider, so it slices latest volume
                node.overlaid_images[overlay_idx].set_data(
                    node.image_funcs[overlay_idx](node.pos)
                )

    # -------------------------
    # internal helpers
    # -------------------------
    def _resolve_clim(self, vol: np.ndarray, clim: Optional[Union[List, Tuple]]):
        if clim is None or clim == 'auto':
            if type(vol) == np.memmap:
                pass
            return utils.auto_clim(vol)
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
        clims = [self._resolve_clim(self.volume, self.base_clim)]
        interps = [self.base_interpolation]

        return AxisAlignedImage(
            image_funcs=image_funcs,
            axis=axis,
            pos=pos,
            limit=limit,
            cmaps=cmaps,
            clims=clims,
            interpolation=interps,
            method=self.method,
            texture_format=self.texture_format,
        )

    def _attach_overlay_to_existing_nodes(self, name: str):
        spec = self._overlays[name]
        for axis in ('x', 'y', 'z'):
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
                    cmap=spec.cmap,
                    clim=list(spec.clim) if spec.clim is not None else list(self._resolve_clim(spec.volume, None)),
                    interpolation=spec.interpolation,
                    method=spec.method,
                    texture_format=spec.texture_format,
                )
                node.overlaid_images.append(overlay)

                overlay_idx = len(node.overlaid_images) - 1
                self._overlay_indices[(axis, i)][name] = overlay_idx

                # initialize overlay data to current slice
                overlay.set_data(node.image_funcs[overlay_idx](node.pos))


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
            )
            line.link_image(z_img)
            line.refresh()
            lines_nodes.append(line)

        return lines_nodes
