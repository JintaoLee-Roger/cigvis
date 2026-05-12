"""
3D vispy canvas using VolumeImage.

Key changes from old gui3d:
  - Uses VolumeImage instead of cigvis.create_slices
  - SamLikeVolumeApp wired in for optional SAM-like interaction
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, List, Dict, Callable, Tuple

import numpy as np

from PySide6.QtWidgets import QWidget, QVBoxLayout, QMessageBox
import cigvis
from cigvis import colormap
from cigvis.vispynodes import VisCanvas
from cigvis.vispynodes.volume_image import VolumeImage
from cigvis.vispynodes.axis_aligned_image import AxisAlignedImage
from cigvis.vispynodes.meshnode import SurfaceNode


_KNOWN_CMAPS = (
    'gray', 'seismic', 'Petrel', 'stratum', 'jet',
    'od_seismic1', 'bwp', 'od_seismic2', 'od_seismic3',
)


def _base_image(node: AxisAlignedImage):
    images = getattr(node, 'overlaid_images', None)
    if images:
        return images[0]
    return node


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


def _guess_cmap_name(image) -> Optional[str]:
    name = getattr(image, '_cigvis_cmap_name', None)
    if isinstance(name, str) and name:
        return name

    cmap = getattr(image, 'cmap', None)
    name = getattr(cmap, 'name', None)
    if isinstance(name, str) and name:
        return name

    if cmap is None or not hasattr(cmap, 'colors'):
        return None

    try:
        colors = np.asarray(cmap.colors.rgba)
    except Exception:
        return None

    for candidate in _KNOWN_CMAPS:
        try:
            ref = np.asarray(colormap.cmap_to_vispy(candidate).colors.rgba)
        except Exception:
            continue
        if colors.shape == ref.shape and np.allclose(colors, ref, atol=1e-6):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Mixins
# ---------------------------------------------------------------------------

class BaseVolumeMixin:
    """Manage the base VolumeImage node."""

    def set_base_data(self, data: np.ndarray) -> None:
        if self._vol is not None:
            self.clear()

        self._data = data
        self._vol = VolumeImage(
            data,
            cmap=self._base_params.get('cmap', 'gray'),
            clim=self._base_params.get('clim'),
            interpolation=self._base_params.get('interpolation', 'linear'),
        )

        nx, ny, nz = self._vol.shape
        xi = nx // 2
        yi = ny // 2
        zi = nz // 2
        self._vol.create_slices([xi], [yi], [zi])
        nodes = self._vol.nodes(intersection_lines=True)
        for n in nodes:
            self._nodes.append(n)
        self.canvas.add_nodes(nodes)

    def set_cmap(self, cmap_name: str) -> None:
        try:
            cmap_v = colormap.cmap_to_vispy(cmap_name)
            self._base_params['cmap'] = cmap_name
            self._base_params['cmap_v'] = cmap_v
            for n in self._nodes:
                if isinstance(n, AxisAlignedImage):
                    base = _base_image(n)
                    base.cmap = cmap_v
                    _set_visual_metadata(base, _cigvis_cmap_name=cmap_name)
                    _set_visual_metadata(n, _cigvis_cmap_name=cmap_name)
            self.canvas.update()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Colormap error: {e}")

    def set_vmin(self, vmin_str: str) -> None:
        if not vmin_str:
            return
        vmin = float(vmin_str)
        self._base_params['vmin'] = vmin
        vmax = self._base_params.get('vmax')
        if vmax is not None:
            clim = [vmin, vmax]
            self._base_params['clim'] = clim
            for n in self._nodes:
                if isinstance(n, AxisAlignedImage):
                    _base_image(n).clim = clim
            self.canvas.update()

    def set_vmax(self, vmax_str: str) -> None:
        if not vmax_str:
            return
        vmax = float(vmax_str)
        self._base_params['vmax'] = vmax
        vmin = self._base_params.get('vmin')
        if vmin is not None:
            clim = [vmin, vmax]
            self._base_params['clim'] = clim
            for n in self._nodes:
                if isinstance(n, AxisAlignedImage):
                    _base_image(n).clim = clim
            self.canvas.update()

    def set_interp(self, interp: str) -> None:
        self._base_params['interpolation'] = interp
        for n in self._nodes:
            if isinstance(n, AxisAlignedImage):
                _base_image(n).interpolation = interp
                _set_visual_metadata(n, _cigvis_interpolation=interp)
        self.canvas.update()


class MaskMixin3D:
    """Manage overlay mask volumes via VolumeImage."""

    def add_mask(self, data: np.ndarray) -> None:
        if self._vol is None:
            return
        name = f'mask_{len(self._masks)}'
        params = {
            'cmap': colormap.set_alpha('jet', 0.5),
            'interpolation': 'nearest',
        }
        self._mask_params.append({'name': name, **params})
        self._masks.append(data)
        try:
            self._vol.add_overlay_volume(
                name=name,
                volume=data,
                cmap=params['cmap'],
                interpolation=params['interpolation'],
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Mask error: {e}")

    def set_mask_params(self, params: list) -> None:
        if self._vol is None:
            return
        idx, mode, value = params
        if idx < 0 or idx >= len(self._mask_params):
            return
        mp = self._mask_params[idx]
        name = mp['name']

        if mode in ('vmin', 'vmax'):
            mp[mode] = float(value)
            if 'vmin' in mp and 'vmax' in mp:
                clim = [mp['vmin'], mp['vmax']]
                for n in self._nodes:
                    if isinstance(n, AxisAlignedImage):
                        spec = self._vol._overlays.get(name)
                        if spec:
                            spec.clim = tuple(clim)
                            # trigger re-render via refresh
                            self._vol.refresh_overlay(name)

        elif mode == 'cmap':
            mp['cmap'] = value
            cmap_v = self._build_cmap(value, mp.get('alpha', 0.5), mp.get('except', 'None'))
            if cmap_v:
                mp['cmap_v'] = cmap_v
                spec = self._vol._overlays.get(name)
                if spec:
                    spec.cmap = cmap_v
                    self._vol.refresh_overlay(name)

        elif mode == 'alpha':
            mp['alpha'] = float(value)
            cmap_v = self._build_cmap(mp.get('cmap', 'jet'), float(value), mp.get('except', 'None'))
            if cmap_v:
                mp['cmap_v'] = cmap_v
                spec = self._vol._overlays.get(name)
                if spec:
                    spec.cmap = cmap_v
                    self._vol.refresh_overlay(name)

        elif mode == 'except':
            mp['except'] = value
            cmap_v = self._build_cmap(mp.get('cmap', 'jet'), mp.get('alpha', 0.5), value)
            if cmap_v:
                mp['cmap_v'] = cmap_v
                spec = self._vol._overlays.get(name)
                if spec:
                    spec.cmap = cmap_v
                    self._vol.refresh_overlay(name)

        elif mode == 'interp':
            spec = self._vol._overlays.get(name)
            if spec:
                spec.interpolation = value
                self._vol.refresh_overlay(name)

    def _build_cmap(self, cmap_name, alpha, excpt):
        try:
            if excpt == 'None':
                return colormap.cmap_to_vispy(colormap.set_alpha(cmap_name, alpha))
            elif excpt == 'min':
                return colormap.cmap_to_vispy(colormap.set_alpha_except_min(cmap_name, alpha))
            elif excpt == 'max':
                return colormap.cmap_to_vispy(colormap.set_alpha_except_max(cmap_name, alpha))
            elif excpt == 'ramp':
                return colormap.cmap_to_vispy(colormap.ramp(cmap_name))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Colormap: {e}")
        return None

    def remove_mask(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._masks):
            return
        self._masks.pop(idx)
        self._mask_params.pop(idx)
        # Rebuild overlays (simplest approach for now)
        # TODO: proper remove if VolumeImage gains a remove_overlay method

    def mask_clear(self) -> None:
        self._masks.clear()
        self._mask_params.clear()


class HorizonMixin3D:
    """Manage horizon surface nodes."""

    def add_horizon(self, data: np.ndarray) -> None:
        if self._data is None:
            return
        params = {'values': 'depth', 'cmaps': 'jet', 'offset': [0, 0, 0], 'interval': [1, 1, 1]}
        self._horz_params.append(params)
        self._horzs.append(data)
        node = SurfaceNode(data, self._data, **params)
        self._horz_nodes.append(node)
        self.canvas.add_node(node)

    def set_horz_params(self, params: list) -> None:
        idx, mode, value = params
        if idx < 0 or idx >= len(self._horzs):
            return
        hp = self._horz_params[idx]
        node = self._horz_nodes[idx]

        if mode == 'coord':
            hp['offset'] = value[0]
            hp['interval'] = value[1]
            node.update_offset_and_interval(value[0], value[1])
        elif mode == 'value_type':
            hp['values'] = value
            if value == 'depth':
                node.values = ['depth']
                node._cmaps = [hp['cmaps']]
                node.clims = None
            elif value == 'amp':
                if self._nodes:
                    node.update_colors_by_slice_node(
                        self._nodes, [self._data] + self._masks)
        elif mode == 'cmap':
            hp['cmaps'] = value
            try:
                node.cmaps = [value]
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Cmap: {e}")

    def remove_horizon(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._horzs):
            return
        node = self._horz_nodes.pop(idx)
        self.canvas.remove_node(node)
        self._horzs.pop(idx)
        self._horz_params.pop(idx)

    def horz_clear(self) -> None:
        for i in range(len(self._horzs)):
            self.remove_horizon(0)


class CameraMixin3D:
    """Camera control helpers."""

    def set_azimuth(self, v: int) -> None:
        for view in self.canvas.view:
            view.camera.azimuth = v

    def set_elevation(self, v: int) -> None:
        for view in self.canvas.view:
            view.camera.elevation = v

    def set_fov(self, v: int) -> None:
        for view in self.canvas.view:
            view.camera.fov = v

    def set_xpos(self, pos: int) -> None:
        for n in self._nodes:
            if isinstance(n, AxisAlignedImage) and n.axis == 'x':
                n._update_location(int(pos))
        self.canvas.update()

    def set_ypos(self, pos: int) -> None:
        for n in self._nodes:
            if isinstance(n, AxisAlignedImage) and n.axis == 'y':
                n._update_location(int(pos))
        self.canvas.update()

    def set_zpos(self, pos: int) -> None:
        for n in self._nodes:
            if isinstance(n, AxisAlignedImage) and n.axis == 'z':
                n._update_location(int(pos))
        self.canvas.update()

    def _apply_aspect(self, idx: int, value: float, reversed_flag: bool) -> None:
        value *= (1 - 2 * int(reversed_flag))
        for view in self.canvas.view:
            f = list(view.camera._flip_factors)
            f[idx] = value
            view.camera._flip_factors = f
            view.camera._update_camera_pos()
        self.canvas.update()

    def set_aspectx(self, v: float) -> None:
        self._apply_aspect(0, v, cigvis.is_x_reversed())

    def set_aspecty(self, v: float) -> None:
        self._apply_aspect(1, v, cigvis.is_y_reversed())

    def set_aspectz(self, v: float) -> None:
        self._apply_aspect(2, v, cigvis.is_z_reversed())

    def get_camera_params(self) -> Optional[list]:
        if not hasattr(self.canvas, 'view') or not self.canvas.view:
            return None
        cam = self.canvas.view[0].camera
        params = [int(cam.azimuth), int(cam.elevation), int(cam.fov)]
        params.append(self.get_slice_positions())
        return params


class DragDropMixin3D:

    def enable_drop(self) -> None:
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        path = event.mimeData().urls()[0].toLocalFile()
        if path and self.parent() and hasattr(self.parent(), 'on_file_dropped'):
            self.parent().on_file_dropped(path)


# ---------------------------------------------------------------------------
# Main 3D canvas widget
# ---------------------------------------------------------------------------

class PlotCanvas3D(
    QWidget,
    DragDropMixin3D,
    CameraMixin3D,
    BaseVolumeMixin,
    MaskMixin3D,
    HorizonMixin3D,
):
    """
    3D vispy canvas widget using VolumeImage.

    Embed a VisCanvas inside a QWidget so it plays nicely with PySide6 layouts.
    """

    def __init__(
        self,
        parent=None,
        visual_nodes=None,
        grid: Optional[Tuple[int, int]] = None,
        share: bool = False,
        canvas_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._init_state()

        # The VisCanvas instance — start with no nodes; add them after data load
        canvas_kwargs = dict(canvas_kwargs or {})
        canvas_kwargs.setdefault('keys', None)
        self.canvas = VisCanvas(
            visual_nodes=visual_nodes,
            grid=grid,
            share=share,
            size=canvas_kwargs.pop('size', (800, 600)),
            **canvas_kwargs,
        )
        self.canvas.create_native()
        self.canvas.native.setParent(self)
        self._layout.addWidget(self.canvas.native)

        self.enable_drop()
        if visual_nodes is not None:
            self._adopt_visual_nodes(visual_nodes)

    def _init_state(self) -> None:
        self._data: Optional[np.ndarray] = None
        self._vol: Optional[VolumeImage] = None
        self._nodes: List = []
        self._base_params: dict = {'cmap': 'gray', 'interpolation': 'linear'}

        self._masks: List[np.ndarray] = []
        self._mask_params: List[dict] = []

        self._horzs: List[np.ndarray] = []
        self._horz_params: List[dict] = []
        self._horz_nodes: List = []

        self._plot_nodes = None

    def _flatten_nodes(self, nodes) -> List:
        if nodes is None:
            return []
        if isinstance(nodes, dict):
            out = []
            for value in nodes.values():
                out.extend(self._flatten_nodes(value))
            return out
        if isinstance(nodes, (list, tuple)):
            out = []
            for item in nodes:
                out.extend(self._flatten_nodes(item))
            return out
        return [nodes]

    def _adopt_visual_nodes(self, nodes) -> None:
        self._plot_nodes = nodes
        flat_nodes = self._flatten_nodes(nodes)
        self._nodes = [n for n in flat_nodes if isinstance(n, AxisAlignedImage)]
        self._horz_nodes = [n for n in flat_nodes if isinstance(n, SurfaceNode)]
        if self._nodes:
            image = _base_image(self._nodes[0])
            cmap_name = _guess_cmap_name(image)
            if cmap_name:
                self._base_params['cmap'] = cmap_name
            interp = getattr(image, 'interpolation', None)
            if interp:
                self._base_params['interpolation'] = interp
            clim = getattr(image, 'clim', None)
            if clim is not None and len(clim) == 2:
                self._base_params['vmin'] = float(clim[0])
                self._base_params['vmax'] = float(clim[1])
                self._base_params['clim'] = [float(clim[0]), float(clim[1])]

    def get_slice_limits(self) -> Dict[str, Tuple[int, int]]:
        limits: Dict[str, Tuple[int, int]] = {}
        for axis in ('x', 'y', 'z'):
            axis_nodes = [
                n for n in self._nodes
                if isinstance(n, AxisAlignedImage) and n.axis == axis and n.limit is not None
            ]
            if axis_nodes:
                lo = min(int(n.limit[0]) for n in axis_nodes)
                hi = max(int(n.limit[1]) for n in axis_nodes)
                limits[axis] = (lo, hi)
        return limits

    def get_slice_positions(self) -> Dict[str, int]:
        positions: Dict[str, int] = {}
        for axis in ('x', 'y', 'z'):
            axis_nodes = [
                n for n in self._nodes
                if isinstance(n, AxisAlignedImage) and n.axis == axis
            ]
            if axis_nodes:
                positions[axis] = int(axis_nodes[0].pos)
        return positions

    def get_base_display_params(self) -> Dict[str, Any]:
        params = dict(self._base_params)
        if self._nodes:
            image = _base_image(self._nodes[0])
            cmap_name = _guess_cmap_name(image)
            if cmap_name:
                params['cmap'] = cmap_name
            interp = getattr(image, 'interpolation', None)
            if interp:
                params['interpolation'] = interp
            clim = getattr(image, 'clim', None)
            if clim is not None and len(clim) == 2:
                params['clim'] = [float(clim[0]), float(clim[1])]
                params['vmin'] = float(clim[0])
                params['vmax'] = float(clim[1])
        return params

    def clear(self) -> None:
        self.horz_clear()
        self.mask_clear()
        for n in self._nodes:
            try:
                n.parent = None
            except Exception:
                pass
        self._nodes.clear()
        try:
            if hasattr(self._data, 'close'):
                self._data.close()
        except Exception:
            pass
        self._data = None
        self._vol = None
        self._base_params = {'cmap': 'gray', 'interpolation': 'linear'}
