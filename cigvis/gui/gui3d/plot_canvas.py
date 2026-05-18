"""
3D vispy canvas using VolumeImage.

Key changes from old gui3d:
  - Uses VolumeImage instead of cigvis.create_slices
  - Prompt-decoder interaction can be wired in by the GUI shell
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, List, Dict, Callable, Tuple

import numpy as np

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QWidget, QVBoxLayout, QMessageBox
import cigvis
from cigvis import colormap
from cigvis.vispynodes import VisCanvas
from cigvis.vispynodes.volume_image import VolumeImage
from cigvis.vispynodes.axis_aligned_image import AxisAlignedImage
from cigvis.vispynodes.meshnode import SurfaceNode
from cigvis.vispynodes.colorbar import Colorbar
from cigvis.vispynodes.splat import Splat
from cigvis.gui.cmap_utils import resolve_gui_cmap


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
    return _guess_cmap_name_from_cmap(cmap)


def _guess_cmap_name_from_cmap(cmap) -> Optional[str]:
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


def _guess_cmap_alpha(cmap, default: float = 0.5) -> float:
    try:
        colors = np.asarray(cmap.colors.rgba)
        alphas = colors[:, 3]
        alphas = alphas[np.isfinite(alphas)]
        alphas = alphas[alphas > 1e-6]
        if alphas.size:
            return float(np.round(np.median(alphas), 3))
    except Exception:
        pass
    return default


def _source_matches(source, select: str, idx: int = 0, idx2: int = 0) -> bool:
    if not isinstance(source, dict):
        return False
    if source.get('select') != select:
        return False
    if int(source.get('idx', 0)) != int(idx):
        return False
    if int(source.get('idx2', 0)) != int(idx2):
        return False
    return True


# ---------------------------------------------------------------------------
# Mixins
# ---------------------------------------------------------------------------

class BaseVolumeMixin:
    """Manage the base VolumeImage node."""

    def _samples_hint(self, source=None) -> int:
        if source is None:
            if self._vol is not None:
                shape = getattr(self._vol, 'shape', None)
            else:
                shape = getattr(self._data, 'shape', None)
        else:
            shape = getattr(source, 'shape', None)
        if shape is not None and len(shape) >= 3:
            return int(shape[2])
        return 256

    def _sync_colorbars(
        self,
        select: str,
        idx: int = 0,
        idx2: int = 0,
        cmap: Any = None,
        clim: Any = None,
        preserve_alpha: Optional[bool] = None,
    ) -> bool:
        params: Dict[str, Any] = {}
        if cmap is not None:
            params['cmap'] = cmap
        if clim is not None:
            params['clim'] = list(clim)
        if preserve_alpha is not None:
            params['preserve_alpha'] = bool(preserve_alpha)
        if not params:
            return False

        updated = False
        for cbar in self._cbar_nodes:
            source = getattr(cbar, '_cigvis_cbar_source', None)
            if not _source_matches(source, select, idx, idx2):
                continue
            if getattr(cbar, 'cbar_size', None) is None:
                continue
            try:
                cbar.update_params(**params)
                updated = True
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Colorbar update error: {e}")
        return updated

    def _sync_surface_colorbars(self, idx: int) -> bool:
        if idx < 0 or idx >= len(self._horz_nodes):
            return False
        node = self._horz_nodes[idx]
        updated = False
        for cbar in self._cbar_nodes:
            source = getattr(cbar, '_cigvis_cbar_source', None)
            if not isinstance(source, dict) or source.get('select') != 'surface':
                continue
            if int(source.get('idx', 0)) != int(idx):
                continue
            idx2 = int(source.get('idx2', 0))
            try:
                cmap = node.cmaps[idx2]
                clim = node.clims[idx2]
            except Exception:
                continue
            updated |= self._sync_colorbars(
                'surface',
                idx=idx,
                idx2=idx2,
                cmap=cmap,
                clim=clim,
            )
        return updated

    def set_base_data(self,
                      data: np.ndarray,
                      display_range: Optional[Dict[str, Tuple[int, int]]] = None
                      ) -> None:
        old_data = self._data
        if self._vol is not None:
            self.clear(close_data=data is not old_data)

        self._data = data
        self._base_params['display_range'] = display_range
        cmap_name = self._base_params.get('cmap', 'gray')
        resolved = resolve_gui_cmap(
            cmap_name,
            samples_hint=self._samples_hint(data),
        )
        self._base_params['cmap_v'] = resolved.cmap
        self._vol = VolumeImage(
            data,
            cmap=resolved.cmap,
            clim=self._base_params.get('clim'),
            interpolation=self._base_params.get('interpolation', 'linear'),
            display_range=display_range,
        )

        ranges = self._vol.display_range
        xi = (ranges['x'][0] + ranges['x'][1] - 1) // 2
        yi = (ranges['y'][0] + ranges['y'][1] - 1) // 2
        zi = (ranges['z'][0] + ranges['z'][1] - 1) // 2
        self._vol.create_slices([xi], [yi], [zi])
        nodes = self._vol.nodes(intersection_lines=True)
        for n in nodes:
            self._nodes.append(n)
            if isinstance(n, AxisAlignedImage):
                base = _base_image(n)
                _set_visual_metadata(base, _cigvis_cmap_name=cmap_name)
                _set_visual_metadata(n, _cigvis_cmap_name=cmap_name)
        self.canvas.add_nodes(nodes)

    def set_cmap(self, cmap_name: str) -> None:
        try:
            resolved = resolve_gui_cmap(
                cmap_name,
                samples_hint=self._samples_hint(),
            )
            cmap_v = resolved.cmap
            self._base_params['cmap'] = cmap_name
            self._base_params['cmap_v'] = cmap_v
            for n in self._nodes:
                if isinstance(n, AxisAlignedImage):
                    base = _base_image(n)
                    base.cmap = cmap_v
                    _set_visual_metadata(base, _cigvis_cmap_name=cmap_name)
                    _set_visual_metadata(n, _cigvis_cmap_name=cmap_name)
            self._sync_colorbars(
                'slices',
                cmap=cmap_v,
                preserve_alpha=resolved.is_line_cmap,
            )
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
            self._sync_colorbars('slices', clim=clim)
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
            self._sync_colorbars('slices', clim=clim)
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
        cmap_name = 'jet'
        alpha = 0.5
        cmap_v = colormap.set_alpha(cmap_name, alpha)
        params = {
            'cmap': cmap_name,
            'cmap_v': cmap_v,
            'alpha': alpha,
            'except': 'None',
            'interpolation': 'nearest',
        }
        self._mask_params.append({'name': name, **params})
        self._masks.append(data)
        try:
            self._vol.add_overlay_volume(
                name=name,
                volume=data,
                cmap=cmap_v,
                cmap_names=cmap_name,
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
            if value in (None, ''):
                return
            mp[mode] = float(value)
            if 'vmin' in mp and 'vmax' in mp:
                clim = [mp['vmin'], mp['vmax']]
                mp['clim'] = clim
                spec = self._vol._overlays.get(name)
                if spec:
                    spec.clim = tuple(clim)
                    self._vol.refresh_overlay(name)
                    self._sync_colorbars('mask', idx=idx, cmap=spec.cmap_for_axis('x'), clim=clim)
                    self.canvas.update()

        elif mode == 'cmap':
            mp['cmap'] = value
            cmap_res = self._resolve_mask_cmap(
                value,
                mp.get('alpha', 0.5),
                mp.get('except', 'None'),
                samples_hint=self._samples_hint(self._masks[idx]),
            )
            cmap_v = cmap_res.cmap if cmap_res else None
            if cmap_v:
                mp['cmap_v'] = cmap_v
                spec = self._vol._overlays.get(name)
                if spec:
                    spec.cmap = cmap_v
                    spec.cmap_name = value
                    spec.axis_cmaps = None
                    spec.axis_cmap_names = None
                    self._vol.refresh_overlay(name)
                    self._sync_colorbars(
                        'mask',
                        idx=idx,
                        cmap=cmap_v,
                        clim=spec.clim,
                        preserve_alpha=cmap_res.is_line_cmap,
                    )
                    self.canvas.update()

        elif mode == 'alpha':
            mp['alpha'] = float(value)
            cmap_res = self._resolve_mask_cmap(
                mp.get('cmap', 'jet'),
                float(value),
                mp.get('except', 'None'),
                samples_hint=self._samples_hint(self._masks[idx]),
            )
            cmap_v = cmap_res.cmap if cmap_res else None
            if cmap_v:
                mp['cmap_v'] = cmap_v
                spec = self._vol._overlays.get(name)
                if spec:
                    spec.cmap = cmap_v
                    spec.cmap_name = mp.get('cmap', 'jet')
                    spec.axis_cmaps = None
                    spec.axis_cmap_names = None
                    self._vol.refresh_overlay(name)
                    self._sync_colorbars(
                        'mask',
                        idx=idx,
                        cmap=cmap_v,
                        clim=spec.clim,
                        preserve_alpha=cmap_res.is_line_cmap,
                    )
                    self.canvas.update()

        elif mode == 'except':
            mp['except'] = value
            cmap_res = self._resolve_mask_cmap(
                mp.get('cmap', 'jet'),
                mp.get('alpha', 0.5),
                value,
                samples_hint=self._samples_hint(self._masks[idx]),
            )
            cmap_v = cmap_res.cmap if cmap_res else None
            if cmap_v:
                mp['cmap_v'] = cmap_v
                spec = self._vol._overlays.get(name)
                if spec:
                    spec.cmap = cmap_v
                    spec.cmap_name = mp.get('cmap', 'jet')
                    spec.axis_cmaps = None
                    spec.axis_cmap_names = None
                    self._vol.refresh_overlay(name)
                    self._sync_colorbars(
                        'mask',
                        idx=idx,
                        cmap=cmap_v,
                        clim=spec.clim,
                        preserve_alpha=cmap_res.is_line_cmap,
                    )
                    self.canvas.update()

        elif mode == 'interp':
            spec = self._vol._overlays.get(name)
            if spec:
                mp['interpolation'] = value
                spec.interpolation = value
                self._vol.refresh_overlay(name)
                self.canvas.update()

    def _resolve_mask_cmap(self, cmap_name, alpha, excpt, samples_hint=None):
        try:
            return resolve_gui_cmap(
                cmap_name,
                alpha=alpha,
                excpt=excpt,
                samples_hint=samples_hint,
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Colormap: {e}")
        return None

    def _build_cmap(self, cmap_name, alpha, excpt, samples_hint=None):
        resolved = self._resolve_mask_cmap(cmap_name, alpha, excpt, samples_hint)
        return resolved.cmap if resolved else None

    def remove_mask(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._masks):
            return
        name = self._mask_params[idx].get('name')
        if self._vol is not None and name:
            try:
                self._vol.remove_overlay(name)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Remove mask error: {e}")
                return
        self._masks.pop(idx)
        self._mask_params.pop(idx)
        self.canvas.update()

    def mask_clear(self) -> None:
        if self._vol is not None:
            for params in list(self._mask_params):
                name = params.get('name')
                if name:
                    try:
                        self._vol.remove_overlay(name)
                    except Exception:
                        pass
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
            self._sync_surface_colorbars(idx)
            self.canvas.update()
        elif mode == 'cmap':
            hp['cmaps'] = value
            try:
                resolved = resolve_gui_cmap(
                    value,
                    samples_hint=self._samples_hint(),
                )
                node.cmaps = [resolved.cmap]
                self._sync_colorbars(
                    'surface',
                    idx=idx,
                    cmap=resolved.cmap,
                    preserve_alpha=resolved.is_line_cmap,
                )
                self.canvas.update()
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


class SplatMixin3D:
    """Manage Gaussian splat overlay nodes."""

    def add_splat(self, data: np.ndarray, params: Optional[dict] = None) -> bool:
        if self._vol is None or not getattr(self.canvas, 'view', None):
            QMessageBox.critical(self, "Error", "Load base data before adding splats.")
            return False
        params = dict(params or {})
        name = params.get('name') or f'splat_{len(self._splat_nodes)}'
        try:
            pos, values = self._splat_payload(
                data,
                threshold=float(params.get('threshold', 0.0)),
            )
            if len(pos) == 0:
                QMessageBox.warning(self, "Empty", "No non-zero splat points found.")
                return False
            from cigvis.vispyplot import create_splats
            nodes = create_splats(
                pos,
                values=values,
                cmap=params.get('cmap', 'viridis'),
                size=float(params.get('size', 3.0)),
                mode=params.get('mode', 'volume'),
                alpha=float(params.get('alpha', 0.45)),
                max_points=params.get('max_points'),
            )
            if not nodes:
                QMessageBox.warning(self, "Empty", "No splat node was created.")
                return False
            node = nodes[0]
            self.canvas.add_node(node)
            self._splat_nodes.append(node)
            self._splat_params.append({'name': name, **params})
            self.canvas.update()
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Splat error: {e}")
            return False

    def _splat_payload(self, data: np.ndarray, threshold: float = 0.0):
        arr = np.asarray(data)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            pos = arr[:, :3].astype(np.float32, copy=False)
            values = arr[:, 3].astype(np.float32, copy=False) if arr.shape[1] > 3 else None
            return pos, values
        if arr.ndim != 3:
            raise ValueError("Splat data must be a 3D mask volume or an (N,3)/(N,4) array.")

        display_range = self._vol.display_range if self._vol is not None else {
            'x': (0, arr.shape[0]),
            'y': (0, arr.shape[1]),
            'z': (0, arr.shape[2]),
        }
        xs, xe = display_range['x']
        ys, ye = display_range['y']
        zs, ze = display_range['z']
        sub = np.asarray(arr[xs:xe, ys:ye, zs:ze])
        finite = np.isfinite(sub)
        if threshold > 0:
            mask = finite & (np.abs(sub) > threshold)
        else:
            mask = finite & (sub != 0)
        coords = np.argwhere(mask)
        if coords.size == 0:
            return np.empty((0, 3), dtype=np.float32), None
        offsets = np.array([xs, ys, zs], dtype=np.float32)
        pos = coords.astype(np.float32) + offsets
        values = sub[mask].astype(np.float32, copy=False)
        return pos, values

    def set_splat_params(self, params: list) -> None:
        idx, mode, value = params
        if idx < 0 or idx >= len(self._splat_nodes):
            return
        node = self._splat_nodes[idx]
        sp = self._splat_params[idx]
        try:
            if mode == 'alpha':
                sp['alpha'] = float(value)
                node.alpha = float(value)
            elif mode == 'visible':
                sp['visible'] = bool(value)
                node.visible = bool(value)
            elif mode == 'size':
                sp['size'] = float(value)
                data = getattr(node, '_data', None)
                if data is not None:
                    node.set_data(
                        pos=data['a_position'],
                        size=float(value),
                        color=data['a_color'],
                    )
            self.canvas.update()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Splat update error: {e}")

    def remove_splat(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._splat_nodes):
            return
        node = self._splat_nodes.pop(idx)
        self._splat_params.pop(idx)
        try:
            self.canvas.remove_node(node)
        except Exception:
            node.parent = None
        self.canvas.update()

    def splat_clear(self) -> None:
        while self._splat_nodes:
            self.remove_splat(0)


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
        native = getattr(getattr(self, 'canvas', None), 'native', None)
        if native is not None:
            native.setAcceptDrops(True)
            native.installEventFilter(self)

    def dragEnterEvent(self, event) -> None:
        self._accept_file_drag(event)

    def dragMoveEvent(self, event) -> None:
        self._accept_file_drag(event)

    def dropEvent(self, event) -> None:
        self._handle_file_drop(event)

    def eventFilter(self, obj, event):
        native = getattr(getattr(self, 'canvas', None), 'native', None)
        if obj is native:
            if event.type() in (QEvent.DragEnter, QEvent.DragMove):
                return self._accept_file_drag(event)
            if event.type() == QEvent.Drop:
                return self._handle_file_drop(event)
        return super().eventFilter(obj, event)

    def _accept_file_drag(self, event) -> bool:
        if self._local_file_from_drop(event):
            event.acceptProposedAction()
            return True
        event.ignore()
        return False

    def _handle_file_drop(self, event) -> bool:
        path = self._local_file_from_drop(event)
        if not path:
            event.ignore()
            return False
        target = self.window()
        if target is not None and hasattr(target, 'on_file_dropped'):
            target.on_file_dropped(path)
            event.acceptProposedAction()
            return True
        event.ignore()
        return False

    def _local_file_from_drop(self, event) -> Optional[str]:
        mime = event.mimeData()
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                if path:
                    return path
        return None


# ---------------------------------------------------------------------------
# Main 3D canvas widget
# ---------------------------------------------------------------------------

class PlotCanvas3D(
    DragDropMixin3D,
    QWidget,
    CameraMixin3D,
    BaseVolumeMixin,
    MaskMixin3D,
    HorizonMixin3D,
    SplatMixin3D,
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
        self._splat_nodes: List[Splat] = []
        self._splat_params: List[dict] = []
        self._cbar_nodes: List[Colorbar] = []

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

    def _volume_image_managers(self, nodes) -> List[VolumeImage]:
        managers = []
        for node in self._flatten_nodes(nodes):
            vi = getattr(node, '_cigvis_volume_image', None)
            if vi is None:
                continue
            if not any(vi is manager for manager in managers):
                managers.append(vi)
        return managers

    def _overlay_image_for_name(self, name: str):
        if self._vol is None:
            return None
        for axis in ('x', 'y', 'z'):
            for i, node in enumerate(getattr(self._vol, '_slices', {}).get(axis, [])):
                overlay_idx = getattr(self._vol, '_overlay_indices', {}).get((axis, i), {}).get(name)
                images = getattr(node, 'overlaid_images', [])
                if overlay_idx is not None and overlay_idx < len(images):
                    return images[overlay_idx]
        return None

    def _sync_mask_params_from_volume_image(self) -> None:
        self._masks.clear()
        self._mask_params.clear()
        if self._vol is None:
            return
        for idx, (name, spec) in enumerate(getattr(self._vol, '_overlays', {}).items()):
            if spec.volume is None:
                continue
            image = self._overlay_image_for_name(name)
            axis = spec.axes[0] if spec.axes else 'x'
            cmap_name = spec.cmap_name_for_axis(axis)
            if not isinstance(cmap_name, str) or not cmap_name:
                cmap_name = _guess_cmap_name(image) if image is not None else None
            if not isinstance(cmap_name, str) or not cmap_name:
                cmap_name = _guess_cmap_name_from_cmap(spec.cmap_for_axis(axis))
            clim = spec.clim
            if clim is None and image is not None:
                clim = getattr(image, 'clim', None)
            params = {
                'name': name or f'mask_{idx}',
                'cmap': cmap_name or 'jet',
                'alpha': _guess_cmap_alpha(spec.cmap_for_axis(axis)),
                'except': 'None',
                'interpolation': spec.interpolation,
            }
            if clim is not None and len(clim) == 2:
                params['clim'] = [float(clim[0]), float(clim[1])]
                params['vmin'] = float(clim[0])
                params['vmax'] = float(clim[1])
            self._masks.append(spec.volume)
            self._mask_params.append(params)

    def _adopt_visual_nodes(self, nodes) -> None:
        self._plot_nodes = nodes
        flat_nodes = self._flatten_nodes(nodes)
        self._nodes = [n for n in flat_nodes if isinstance(n, AxisAlignedImage)]
        self._horz_nodes = [n for n in flat_nodes if isinstance(n, SurfaceNode)]
        self._cbar_nodes = [n for n in flat_nodes if isinstance(n, Colorbar)]
        managers = self._volume_image_managers(nodes)
        if len(managers) == 1:
            self._vol = managers[0]
            self._data = self._vol.volume
            self._sync_mask_params_from_volume_image()
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

    def get_mask_display_params(self) -> List[Dict[str, Any]]:
        return [dict(params) for params in self._mask_params]

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

    def clear(self, close_data: bool = True) -> None:
        self.splat_clear()
        self.horz_clear()
        self.mask_clear()
        for n in self._nodes:
            try:
                n.parent = None
            except Exception:
                pass
        self._nodes.clear()
        if close_data:
            try:
                if hasattr(self._data, 'close'):
                    self._data.close()
            except Exception:
                pass
        self._data = None
        self._vol = None
        self._cbar_nodes.clear()
        self._splat_nodes.clear()
        self._splat_params.clear()
        self._base_params = {'cmap': 'gray', 'interpolation': 'linear'}
