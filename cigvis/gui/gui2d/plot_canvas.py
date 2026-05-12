"""
2D plot canvas based on Matplotlib embedded in PySide6.

Supports:
  - 2D data: displayed as single imshow
  - 3D data: treated as a stack of 2D slices (navigate via axis+index)
  - Overlay masks
  - Annotation (point / box / brush)
"""

from __future__ import annotations

from pathlib import Path
from argparse import Namespace
from typing import Optional, List

import numpy as np

from PySide6.QtWidgets import QWidget, QVBoxLayout, QFileDialog, QMessageBox
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from cigvis import colormap
from cigvis.gui.widgets.common import RectP


# ---------------------------------------------------------------------------
# Mixins
# ---------------------------------------------------------------------------

class ImageMixin:

    def set_base_data(self, data: np.ndarray) -> None:
        """Accept 2D or 3D data. 3D is stored; display is driven by set_slice."""
        self._data_full = data
        if data.ndim == 2:
            self._is_3d = False
            self.data = data
            self._plot()
        else:
            self._is_3d = True
            self._slice_axis = 0
            self._slice_idx = 0
            self.data = data[0]  # first slice
            self._plot()

    def set_slice(self, axis: int, idx: int) -> None:
        """Show a specific 2D slice from 3D data."""
        if not self._is_3d or self._data_full is None:
            return
        self._slice_axis = axis
        self._slice_idx = idx
        idx = max(0, min(idx, self._data_full.shape[axis] - 1))
        if axis == 0:
            self.data = self._data_full[idx]
        elif axis == 1:
            self.data = self._data_full[:, idx, :]
        else:
            self.data = self._data_full[:, :, idx]
        self._plot()

    def set_cmap(self, cmap_name: str) -> None:
        try:
            self.params['cmap'] = colormap.get_cmap_from_str(cmap_name)
            if self.baseim is not None:
                self.baseim.set_cmap(self.params['cmap'])
                self.draw()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Colormap error: {e}")

    def set_interp(self, interp: str) -> None:
        self.params['interpolation'] = interp
        if self.baseim is not None:
            self.baseim.set_interpolation(interp)
            self.draw()

    def set_vmin(self, vmin_str: str) -> None:
        if not vmin_str:
            return
        vmin = float(vmin_str)
        self.params['vmin'] = vmin
        vmax = self.params.get('vmax', -vmin)
        if vmin > vmax:
            QMessageBox.critical(self, "Error", "vmin > vmax")
            return
        if self.baseim is not None:
            self.baseim.set(clim=[vmin, vmax])
            self.draw()

    def set_vmax(self, vmax_str: str) -> None:
        if not vmax_str:
            return
        vmax = float(vmax_str)
        self.params['vmax'] = vmax
        vmin = self.params.get('vmin', -vmax)
        if vmin > vmax:
            QMessageBox.critical(self, "Error", "vmax < vmin")
            return
        if self.baseim is not None:
            self.baseim.set(clim=[vmin, vmax])
            self.draw()

    def image_clear(self) -> None:
        self._data_full = None
        self._is_3d = False
        self.data = None
        self.params = {'cmap': 'gray', 'interpolation': 'bilinear'}
        if self.baseim is not None:
            self.baseim.remove()
        self.baseim = None


class MaskMixin:

    def add_mask(self, data: np.ndarray) -> None:
        if len(self.mask_params) == len(self.masks):
            cmap = colormap.set_alpha('jet', 0.5, as_mpl=True)
            self.mask_params.append({
                'mpl': {'interpolation': 'nearest', 'cmap': cmap},
                'alpha': 0.5, 'except': 'None', 'cmap': 'jet',
            })
        self.masks.append(data)
        im = self.axes.imshow(data.T, **self.mask_params[-1]['mpl'])
        self.maskim.append(im)
        self.draw()

    def set_mask_params(self, params: list) -> None:
        idx, mode, value = params
        if idx < 0 and len(self.mask_params) == len(self.masks):
            cmap = colormap.set_alpha('jet', 0.5, as_mpl=True)
            self.mask_params.append({
                'mpl': {'interpolation': 'nearest', 'cmap': cmap},
                'alpha': 0.5, 'except': 'None', 'cmap': 'jet',
            })

        mp = self.mask_params[idx]
        if mode == 'vmin':
            mp['mpl']['vmin'] = float(value)
        elif mode == 'vmax':
            mp['mpl']['vmax'] = float(value)
        elif mode == 'cmap':
            mp['cmap'] = value
            c = self._build_mask_cmap(value, mp['alpha'], mp['except'])
            if c:
                mp['mpl']['cmap'] = c
                if len(self.mask_params) == len(self.masks):
                    self.maskim[idx].set_cmap(c)
        elif mode == 'interp':
            mp['mpl']['interpolation'] = value
            if len(self.mask_params) == len(self.masks):
                self.maskim[idx].set_interpolation(value)
        elif mode == 'alpha':
            mp['alpha'] = float(value)
            c = self._build_mask_cmap(mp['cmap'], float(value), mp['except'])
            if c:
                mp['mpl']['cmap'] = c
                if len(self.mask_params) == len(self.masks):
                    self.maskim[idx].set_cmap(c)
        elif mode == 'except':
            mp['except'] = value
            c = self._build_mask_cmap(mp['cmap'], mp['alpha'], value)
            if c:
                mp['mpl']['cmap'] = c
                if len(self.mask_params) == len(self.masks):
                    self.maskim[idx].set_cmap(c)
        self.draw()

    def _build_mask_cmap(self, cmap_name, alpha, excpt):
        try:
            if excpt == 'None':
                return colormap.set_alpha(cmap_name, alpha, as_mpl=True)
            elif excpt == 'min':
                return colormap.set_alpha_except_min(cmap_name, alpha, as_mpl=True)
            elif excpt == 'max':
                return colormap.set_alpha_except_max(cmap_name, alpha, as_mpl=True)
            elif excpt == 'ramp':
                return colormap.ramp(cmap_name, as_mpl=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Colormap error: {e}")
        return None

    def remove_mask(self, idx: int) -> None:
        im = self.maskim.pop(idx)
        im.remove()
        self.mask_params.pop(idx)
        self.masks.pop(idx)
        self.draw()

    def mask_clear(self) -> None:
        for im in self.maskim:
            im.remove()
        self.maskim.clear()
        self.mask_params.clear()
        self.masks.clear()


class AnnotationMixin:

    def set_marker_mode(self, mode: int) -> None:
        self.marker_mode = mode

    def set_box_mode(self, mode: int) -> None:
        self.box_mode = mode

    def set_brush_mode(self, mode: int) -> None:
        self.brush_mode = mode

    def set_brush_size(self, size: int) -> None:
        self.brush_size = size

    def on_mouse_press(self, event) -> None:
        if not event.inaxes:
            return
        if self.marker_mode >= 0:
            c = '#74ddd0' if self.marker_mode == 1 else '#f5c2cb'
            im = self.axes.plot(event.xdata, event.ydata, 'o', color=c)
            self.marker_im.append(im[0])
            self.marker_list.append([event.xdata, event.ydata, self.marker_mode - 1])
            self.draw()
        elif self.box_mode > 0:
            self.box_list.append(RectP(event.xdata, event.ydata, 0, 0))
            self._rect = Rectangle((event.xdata, event.ydata), 0, 0,
                                    fill=False, color='white')
            self.box_im.append(self.axes.add_patch(self._rect))
            self.draw()
        elif self.brush_mode > 0:
            self._scribing = True
            self.scrib_list.append([[event.xdata], [event.ydata]])
            lw = self.brush_size / 10
            self.scribim_list.append(
                self.axes.plot(self.scrib_list[-1][0], self.scrib_list[-1][1],
                               color='red', linewidth=lw)[0])
            self.draw()

    def on_mouse_move(self, event) -> None:
        if self.box_mode > 0 and self._rect is not None and event.inaxes:
            self.box_list[-1].add_p1(event.xdata, event.ydata)
            self._rect.set_width(event.xdata - self.box_list[-1].x0)
            self._rect.set_height(event.ydata - self.box_list[-1].y0)
            self.draw()
        elif self.brush_mode > 0 and self._scribing and event.inaxes:
            self.scrib_list[-1][0].append(event.xdata)
            self.scrib_list[-1][1].append(event.ydata)
            self.scribim_list[-1].set_data(self.scrib_list[-1][0], self.scrib_list[-1][1])
            self.draw()

        if event.inaxes and self.data is not None:
            try:
                x, y = round(event.xdata), round(event.ydata)
                if 0 <= x < self.data.shape[0] and 0 <= y < self.data.shape[1]:
                    v = self.data[x, y]
                    self._status_cb(f"x={event.xdata:.1f}  y={event.ydata:.1f}  v={v:.4g}")
            except Exception:
                pass

    def on_mouse_release(self, event) -> None:
        if self.box_mode > 0:
            self._rect = None
        elif self.brush_mode > 0:
            self._scribing = False

    def on_leave(self, event) -> None:
        if self.box_mode > 0 or self.brush_mode > 0:
            self._scribing = False
            self._rect = None

    def do_marker_undo(self) -> None:
        if self.marker_list:
            self.marker_list.pop()
            self.marker_im.pop().remove()
            self.draw()

    def do_box_undo(self) -> None:
        if self.box_list:
            self.box_list.pop()
            self.box_im.pop().remove()
            self.draw()

    def do_brush_undo(self) -> None:
        if self.scrib_list:
            self.scrib_list.pop()
            self.scribim_list.pop().remove()
            self.draw()

    def do_marker_reset(self) -> None:
        self.marker_list.clear()
        for im in self.marker_im:
            im.remove()
        self.marker_im.clear()
        self.draw()

    def do_box_reset(self) -> None:
        self.box_list.clear()
        for im in self.box_im:
            im.remove()
        self.box_im.clear()
        self.draw()

    def do_brush_reset(self) -> None:
        self.scrib_list.clear()
        for im in self.scribim_list:
            im.remove()
        self.scribim_list.clear()
        self.draw()


class DragDropMixin:

    def enable_drop(self) -> None:
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        path = event.mimeData().urls()[0].toLocalFile()
        if path:
            self._on_drop(path)

    def _on_drop(self, path: str) -> None:
        pass  # override in PlotCanvas


# ---------------------------------------------------------------------------
# Main canvas widget
# ---------------------------------------------------------------------------

class PlotCanvas(FigureCanvas, DragDropMixin, ImageMixin, MaskMixin, AnnotationMixin):
    """Matplotlib canvas that supports 2D/3D-as-2D data with masks + annotation."""

    def __init__(self, parent=None, width=6, height=5, dpi=100) -> None:
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_axis_off()
        self.fig.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.04)
        super().__init__(self.fig)
        self.setParent(parent)

        self.enable_drop()
        self.mpl_connect('button_press_event', self.on_mouse_press)
        self.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.mpl_connect('button_release_event', self.on_mouse_release)
        self.leaveEvent = self.on_leave
        self._init_state()

    def _init_state(self) -> None:
        # base data
        self.data: Optional[np.ndarray] = None
        self._data_full: Optional[np.ndarray] = None
        self._is_3d = False
        self._slice_axis = 0
        self._slice_idx = 0
        self.params = {'cmap': 'gray', 'interpolation': 'bilinear'}
        self.baseim = None

        # masks
        self.masks: List[np.ndarray] = []
        self.maskim: List = []
        self.mask_params: List[dict] = []

        # annotation
        self.marker_mode = -1
        self.marker_list: List = []
        self.marker_im: List = []
        self.box_mode = -1
        self._rect = None
        self.box_list: List[RectP] = []
        self.box_im: List = []
        self.brush_mode = -1
        self._scribing = False
        self.scrib_list: List = []
        self.scribim_list: List = []
        self.brush_size = 10

        # status callback (set by main window)
        self._status_cb = lambda _: None

    def set_status_callback(self, cb) -> None:
        self._status_cb = cb

    def _plot(self) -> None:
        self.axes.clear()
        self.axes.set_axis_off()
        if self.data is None:
            self.baseim = None
        else:
            self.baseim = self.axes.imshow(self.data.T, **self.params, aspect='auto')
        self.draw()

    def clear(self) -> None:
        self.do_marker_reset()
        self.do_box_reset()
        self.do_brush_reset()
        self.mask_clear()
        self.image_clear()
        self.marker_mode = self.box_mode = self.brush_mode = -1
        self._scribing = False
        self.brush_size = 10
        self._plot()

    def save_fig(self) -> None:
        if self.baseim is None:
            QMessageBox.warning(self, "Warning", "No image to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Figure", "", "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)")
        if path:
            self.fig.savefig(path, bbox_inches='tight', pad_inches=0.01, dpi=300)

    def _on_drop(self, path: str) -> None:
        # Try to notify parent (main window) about the dropped file
        if self.parent() and hasattr(self.parent(), 'on_file_dropped'):
            self.parent().on_file_dropped(path)
