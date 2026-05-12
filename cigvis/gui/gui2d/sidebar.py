"""
Sidebar panels for the 2D viewer.

Each panel is a self-contained QWidget designed to live inside the
SlidingDrawer. Panels emit typed signals; the main window (or controller)
connects them to the canvas.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QPushButton, QFileDialog, QMessageBox, QListWidgetItem,
    QSpinBox, QSlider, QFrame,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QRegularExpressionValidator

from cigvis.gui.widgets.common import (
    LineEdit, SpinBox, DoubleSpinBox, EditableComboBox,
    RadioGroup, ToggleButton, RectP,
    ImageParamsWidget, MaskParamsWidget, ItemsWidget,
    LoadFolderWidget, _format_float,
    INT_RE, FLOAT_RE,
)
from cigvis.gui.widgets.collapsible_section import CollapsibleSection


def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setFrameShadow(QFrame.Sunken)
    return f


# ---------------------------------------------------------------------------
# Load panel
# ---------------------------------------------------------------------------

class LoadPanel(QWidget):
    """File loading + dimension controls for 2-D data (or 3-D treated as 2-D stack)."""

    # emits ndarray when base data loaded
    base_loaded = Signal(object)
    # emits ndarray when mask loaded
    mask_loaded = Signal(object)
    # emits (nx, ny) when dims updated
    dims_updated = Signal(int, int)
    # emits vmin, vmax strings after loading
    vmin_ready = Signal(str)
    vmax_ready = Signal(str)
    # emits QListWidgetItem for a new mask
    mask_item_ready = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._nx: Optional[int] = None
        self._ny: Optional[int] = None
        self._data_loaded = False
        self._load_type = 'base'   # 'base' | 'mask'
        self._transpose = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # --- dimension section ---
        dim_sec = CollapsibleSection("Dimensions", collapsed=False)

        dim_row = QHBoxLayout()
        dim_row.addWidget(QLabel("nx"))
        self.nx_edit = LineEdit()
        self.nx_edit.setValidator(QRegularExpressionValidator(INT_RE, self))
        dim_row.addWidget(self.nx_edit)
        dim_row.addWidget(QLabel("ny"))
        self.ny_edit = LineEdit()
        self.ny_edit.setValidator(QRegularExpressionValidator(INT_RE, self))
        dim_row.addWidget(self.ny_edit)
        dim_sec.add_layout(dim_row)

        trans_row = QHBoxLayout()
        trans_label = QLabel("Transpose")
        self.trans_radio = RadioGroup(['on', 'off'])
        trans_row.addWidget(trans_label)
        trans_row.addWidget(self.trans_radio)
        dim_sec.add_layout(trans_row)
        layout.addWidget(dim_sec)

        # --- folder loader ---
        folder_sec = CollapsibleSection("Folder Browse", collapsed=True)
        self.folder_widget = LoadFolderWidget()
        folder_sec.add(self.folder_widget)
        layout.addWidget(folder_sec)

        # --- load type + buttons ---
        load_sec = CollapsibleSection("Load", collapsed=False)
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type"))
        self.load_type_radio = RadioGroup(['base', 'mask'])
        type_row.addWidget(self.load_type_radio)
        load_sec.add_layout(type_row)

        btn_row = QHBoxLayout()
        self.load_btn = QPushButton("Load File")
        self.clear_btn = QPushButton("Clear")
        self.save_btn = QPushButton("Save")
        btn_row.addWidget(self.load_btn)
        btn_row.addWidget(self.clear_btn)
        btn_row.addWidget(self.save_btn)
        load_sec.add_layout(btn_row)
        layout.addWidget(load_sec)

        self._connect()

    def _connect(self) -> None:
        self.nx_edit.editingFinished.connect(self._update_nx)
        self.ny_edit.editingFinished.connect(self._update_ny)
        self.trans_radio.selection_changed.connect(self._set_transpose)
        self.load_type_radio.selection_changed.connect(self._set_load_type)
        self.load_btn.clicked.connect(self.load_file)
        self.folder_widget.current_path.connect(lambda p: self.load_file(p, check=False))

    def _update_nx(self) -> None:
        t = self.nx_edit.text()
        if t:
            self._nx = int(t)

    def _update_ny(self) -> None:
        t = self.ny_edit.text()
        if t:
            self._ny = int(t)

    def _set_transpose(self, text: str) -> None:
        self._transpose = text == 'on'

    def _set_load_type(self, text: str) -> None:
        self._load_type = text

    def load_file(self, file_path: str = "", check: bool = True) -> None:
        # validation
        if check:
            if self._data_loaded and self._load_type == 'base':
                QMessageBox.critical(self, "Error", "Data already loaded. Clear first.")
                return
            if not self._data_loaded and self._load_type == 'mask':
                QMessageBox.critical(self, "Error", "Load base data first.")
                return

        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*)")
        if not file_path:
            return

        try:
            data = self._read_file(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {e}")
            return

        if data is None:
            return

        if self._transpose and data.ndim == 2:
            data = data.T

        if self._load_type == 'base':
            self._data_loaded = True
            if data.ndim == 2:
                nx, ny = data.shape
            else:
                nx, ny = data.shape[1], data.shape[2]
            self._nx, self._ny = nx, ny
            self.nx_edit.set_value(str(nx))
            self.ny_edit.set_value(str(ny))
            from cigvis.utils import utils
            v1, v2 = utils.auto_clim(data)
            self.vmin_ready.emit(_format_float(v1))
            self.vmax_ready.emit(_format_float(v2))
            self.base_loaded.emit(data)
        else:
            item = QListWidgetItem(Path(file_path).name)
            pw = MaskParamsWidget(mode='2d')
            pw.set_callback = lambda cb: cb  # placeholder
            from cigvis.utils import utils
            v1, v2 = utils.auto_clim(data)
            pw.vmin_edit.set_value(_format_float(v1))
            pw.vmax_edit.set_value(_format_float(v2))
            item.params_widget = pw
            self.mask_item_ready.emit(item)
            self.mask_loaded.emit(data)

    def _read_file(self, path: str) -> Optional[np.ndarray]:
        if path.endswith('.npy'):
            d = np.load(path)
            if d.ndim not in (2, 3):
                QMessageBox.critical(self, "Error", f"Expected 2D or 3D array, got {d.ndim}D.")
                return None
            if d.ndim == 2:
                self._nx, self._ny = d.shape
            return d
        else:
            if self._nx is None or self._ny is None:
                QMessageBox.critical(self, "Error", "Set nx and ny before loading binary file.")
                return None
            return np.fromfile(path, np.float32).reshape(self._nx, self._ny)

    def set_dims(self, nx: int, ny: int) -> None:
        self._nx, self._ny = nx, ny
        self.nx_edit.set_value(str(nx))
        self.ny_edit.set_value(str(ny))

    def clear(self, clear_dims: bool = True) -> None:
        self._data_loaded = False
        self._load_type = 'base'
        self.load_type_radio.set_selection('base')
        self.folder_widget.clear()
        if clear_dims:
            self.nx_edit.clear()
            self.ny_edit.clear()
            self._nx = self._ny = None


# ---------------------------------------------------------------------------
# Display panel (colormap, clim, interp)
# ---------------------------------------------------------------------------

class DisplayPanel(QWidget):
    """Colormap / clim / interpolation controls."""

    cmap_changed = Signal(str)
    vmin_changed = Signal(str)
    vmax_changed = Signal(str)
    interp_changed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        base_sec = CollapsibleSection("Base Image", collapsed=False)
        self.params = ImageParamsWidget(mode='2d')
        base_sec.add(self.params)
        layout.addWidget(base_sec)

        # pass-through connections
        self.params.cmap_changed.connect(self.cmap_changed)
        self.params.vmin_changed.connect(self.vmin_changed)
        self.params.vmax_changed.connect(self.vmax_changed)
        self.params.interp_changed.connect(self.interp_changed)


    def set_vmin(self, v: str) -> None:
        self.params.set_vmin(v)

    def set_vmax(self, v: str) -> None:
        self.params.set_vmax(v)

    def clear(self) -> None:
        self.params.clear()


# ---------------------------------------------------------------------------
# Slice navigation panel (for 3D-as-2D viewing)
# ---------------------------------------------------------------------------

class SliceNavPanel(QWidget):
    """
    Navigation panel for 3D data treated as a stack of 2D slices.

    When the loaded data is 3D with shape (n0, n1, n2), this panel lets
    the user choose which axis to iterate over and which slice to show.
    Useful for imbalanced data like gathers: (32, 4000, 2001).
    """

    slice_changed = Signal(int, int)   # (axis_idx, slice_pos)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._shape = None   # (n0, n1, n2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        nav_sec = CollapsibleSection("3D Slice Navigation", collapsed=False)

        # axis selector
        axis_row = QHBoxLayout()
        axis_row.addWidget(QLabel("Axis"))
        self.axis_radio = RadioGroup(['0', '1', '2'])
        axis_row.addWidget(self.axis_radio)
        nav_sec.add_layout(axis_row)

        # index spinbox + slider
        idx_row = QHBoxLayout()
        idx_row.addWidget(QLabel("Index"))
        self.idx_spin = SpinBox()
        self.idx_spin.setMinimum(0)
        idx_row.addWidget(self.idx_spin)
        nav_sec.add_layout(idx_row)

        self.idx_slider = QSlider(Qt.Horizontal)
        self.idx_slider.setMinimum(0)
        nav_sec.add(self.idx_slider)

        # shape info
        self.info_label = QLabel("No 3D data")
        self.info_label.setAlignment(Qt.AlignCenter)
        nav_sec.add(self.info_label)

        layout.addWidget(nav_sec)
        self.setEnabled(False)

        self.axis_radio.selection_changed.connect(self._on_axis_changed)
        self.idx_spin.changed.connect(self._on_idx_changed)
        self.idx_slider.valueChanged.connect(self._on_slider_changed)

    def set_shape(self, shape: tuple) -> None:
        """Configure for 3D data of given shape."""
        if len(shape) != 3:
            self.setEnabled(False)
            return
        self._shape = shape
        self.info_label.setText(f"Shape: {shape[0]} × {shape[1]} × {shape[2]}")
        self._update_limits()
        self.setEnabled(True)

    def _update_limits(self) -> None:
        if self._shape is None:
            return
        ax = int(self.axis_radio.get_selection())
        n = self._shape[ax]
        self.idx_spin.blockSignals(True)
        self.idx_slider.blockSignals(True)
        self.idx_spin.setRange(0, n - 1)
        self.idx_spin.setValue(0)
        self.idx_slider.setRange(0, n - 1)
        self.idx_slider.setValue(0)
        self.idx_spin.blockSignals(False)
        self.idx_slider.blockSignals(False)

    def _on_axis_changed(self, _text: str) -> None:
        self._update_limits()
        self._emit()

    def _on_idx_changed(self, idx: int) -> None:
        self.idx_slider.blockSignals(True)
        self.idx_slider.setValue(idx)
        self.idx_slider.blockSignals(False)
        self._emit()

    def _on_slider_changed(self, idx: int) -> None:
        self.idx_spin.blockSignals(True)
        self.idx_spin.setValue(idx)
        self.idx_spin.blockSignals(False)
        self._emit()

    def _emit(self) -> None:
        ax = int(self.axis_radio.get_selection())
        idx = self.idx_spin.value()
        self.slice_changed.emit(ax, idx)

    def get_current(self) -> tuple:
        return int(self.axis_radio.get_selection()), self.idx_spin.value()


# ---------------------------------------------------------------------------
# Annotation panel (marker / box / brush)
# ---------------------------------------------------------------------------

class AnnotationPanel(QWidget):
    """
    SAM-like annotation panel: hover points, bounding boxes, free brush.

    Signals
    -------
    marker_mode_changed  : int  (-1=off, 0=neg, 1=pos)
    box_mode_changed     : int  (-1=off, 1=on)
    brush_mode_changed   : int  (-1=off, 1=on)
    brush_size_changed   : int
    marker_reset         : ()
    marker_undo          : ()
    box_reset            : ()
    box_undo             : ()
    brush_reset          : ()
    brush_undo           : ()
    """

    marker_mode_changed = Signal(int)
    box_mode_changed = Signal(int)
    brush_mode_changed = Signal(int)
    brush_size_changed = Signal(int)
    marker_reset = Signal()
    marker_undo = Signal()
    box_reset = Signal()
    box_undo = Signal()
    brush_reset = Signal()
    brush_undo = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # --- Point annotation ---
        pt_sec = CollapsibleSection("Point Prompts", collapsed=False)
        self.hover_btn = ToggleButton("Hover")
        self.hover_pos = ToggleButton("+")
        self.hover_neg = ToggleButton("−")
        pt_row = QHBoxLayout()
        pt_row.addWidget(self.hover_btn)
        pt_row.addWidget(self.hover_pos)
        pt_row.addWidget(self.hover_neg)
        pt_sec.add_layout(pt_row)

        hover_ctrl = QHBoxLayout()
        self.hover_reset_btn = QPushButton("reset")
        self.hover_undo_btn = QPushButton("undo")
        hover_ctrl.addWidget(self.hover_reset_btn)
        hover_ctrl.addWidget(self.hover_undo_btn)
        pt_sec.add_layout(hover_ctrl)
        layout.addWidget(pt_sec)

        # --- Box annotation ---
        box_sec = CollapsibleSection("Box Prompts", collapsed=False)
        self.box_btn = ToggleButton("Box")
        self.box_add = ToggleButton("+")
        box_row = QHBoxLayout()
        box_row.addWidget(self.box_btn)
        box_row.addWidget(self.box_add)
        box_sec.add_layout(box_row)

        box_ctrl = QHBoxLayout()
        self.box_reset_btn = QPushButton("reset")
        self.box_undo_btn = QPushButton("undo")
        box_ctrl.addWidget(self.box_reset_btn)
        box_ctrl.addWidget(self.box_undo_btn)
        box_sec.add_layout(box_ctrl)
        layout.addWidget(box_sec)

        # --- Brush ---
        brush_sec = CollapsibleSection("Brush", collapsed=False)
        self.brush_btn = ToggleButton("Brush")
        brush_sec.add(self.brush_btn)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Size"))
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(1, 100)
        self.brush_slider.setValue(10)
        size_row.addWidget(self.brush_slider)
        brush_sec.add_layout(size_row)

        brush_ctrl = QHBoxLayout()
        self.brush_reset_btn = QPushButton("reset")
        self.brush_undo_btn = QPushButton("undo")
        brush_ctrl.addWidget(self.brush_reset_btn)
        brush_ctrl.addWidget(self.brush_undo_btn)
        brush_sec.add_layout(brush_ctrl)
        layout.addWidget(brush_sec)

        self._connect()

    def _connect(self) -> None:
        self.hover_btn.clicked.connect(self._active_marker)
        self.hover_pos.clicked.connect(self._active_marker)
        self.hover_neg.clicked.connect(self._active_marker)
        self.box_btn.clicked.connect(self._active_box)
        self.box_add.clicked.connect(self._active_box)
        self.brush_btn.clicked.connect(self._active_brush)
        self.brush_slider.valueChanged.connect(self.brush_size_changed)
        self.hover_reset_btn.clicked.connect(self.marker_reset)
        self.hover_undo_btn.clicked.connect(self.marker_undo)
        self.box_reset_btn.clicked.connect(self.box_reset)
        self.box_undo_btn.clicked.connect(self.box_undo)
        self.brush_reset_btn.clicked.connect(self.brush_reset)
        self.brush_undo_btn.clicked.connect(self.brush_undo)

    def _deactivate_box_brush(self) -> None:
        self.box_btn.setChecked(False)
        self.box_add.setChecked(False)
        self.brush_btn.setChecked(False)
        self.box_mode_changed.emit(-1)
        self.brush_mode_changed.emit(-1)

    def _deactivate_hover(self) -> None:
        self.hover_btn.setChecked(False)
        self.hover_pos.setChecked(False)
        self.hover_neg.setChecked(False)
        self.marker_mode_changed.emit(-1)

    def _active_marker(self) -> None:
        if not self.hover_btn.isChecked():
            self.marker_mode_changed.emit(-1)
            return
        self._deactivate_box_brush()
        if self.hover_pos.isChecked():
            self.hover_neg.setChecked(False)
            self.marker_mode_changed.emit(1)
        elif self.hover_neg.isChecked():
            self.hover_pos.setChecked(False)
            self.marker_mode_changed.emit(0)

    def _active_box(self) -> None:
        if not self.box_btn.isChecked():
            self.box_mode_changed.emit(-1)
            return
        self._deactivate_hover()
        self.brush_btn.setChecked(False)
        self.brush_mode_changed.emit(-1)
        if self.box_add.isChecked():
            self.box_mode_changed.emit(1)

    def _active_brush(self) -> None:
        if not self.brush_btn.isChecked():
            self.brush_mode_changed.emit(-1)
            return
        self._deactivate_hover()
        self.box_btn.setChecked(False)
        self.box_add.setChecked(False)
        self.box_mode_changed.emit(-1)
        self.brush_mode_changed.emit(1)

    def clear(self) -> None:
        self.hover_btn.setChecked(False)
        self.hover_pos.setChecked(False)
        self.hover_neg.setChecked(False)
        self.box_btn.setChecked(False)
        self.box_add.setChecked(False)
        self.brush_btn.setChecked(False)
        self.brush_slider.setValue(10)


# ---------------------------------------------------------------------------
# Overlays panel (mask layers)
# ---------------------------------------------------------------------------

class OverlaysPanel(QWidget):
    """List of mask overlay layers with per-layer parameter controls."""

    params_changed = Signal(list)   # [idx, mode, value]
    item_deleted = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        sec = CollapsibleSection("Mask Layers", collapsed=False)
        self.items_widget = ItemsWidget()
        self.items_widget.params_changed.connect(self.params_changed)
        self.items_widget.item_deleted.connect(self.item_deleted)
        sec.add(self.items_widget)
        layout.addWidget(sec)

    def add_item(self, item: QListWidgetItem) -> None:
        self.items_widget.add_item(item)

    def clear(self) -> None:
        self.items_widget.clear()


# ---------------------------------------------------------------------------
# Status bar widget (value readout)
# ---------------------------------------------------------------------------

class StatusWidget(QWidget):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(4, 2, 4, 2)
        self.label = QLabel("")
        self.label.setStyleSheet("font-size: 11px; color: rgba(255,255,255,0.55);")
        row.addWidget(self.label)

    def set_text(self, text: str) -> None:
        self.label.setText(text)
