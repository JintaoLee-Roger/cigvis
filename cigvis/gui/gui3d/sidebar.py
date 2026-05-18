"""
Sidebar panels for the 3D viewer.

Panels:
  - LoadPanel3D    : load volume, set nx/ny/nz
  - DisplayPanel3D : colormap, clim, interpolation
  - CameraPanel    : azimuth, elevation, FOV, aspect ratios
  - SlicesPanel    : x/y/z position of axis-aligned slices
  - OverlaysPanel3D: mask + horizon layers
  - SamPanel       : SAM-like interactive annotation (optional)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QPushButton, QFileDialog, QMessageBox, QListWidgetItem, QFrame,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QRegularExpressionValidator

from cigvis.gui.widgets.common import (
    LineEdit, SpinBox, DoubleSpinBox, EditableComboBox,
    RadioGroup, ToggleButton,
    ImageParamsWidget, MaskParamsWidget, ItemsWidget,
    LoadFolderWidget, _format_float,
    INT_RE, FLOAT_RE,
)
from cigvis.gui.widgets.collapsible_section import CollapsibleSection


def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    return f


# ---------------------------------------------------------------------------
# Load panel
# ---------------------------------------------------------------------------

class LoadPanel3D(QWidget):
    """File loading for 3D seismic volumes."""

    base_loaded = Signal(object)      # ndarray
    mask_loaded = Signal(object)      # ndarray
    horz_loaded = Signal(object)      # ndarray (2D horizon)
    vmin_ready = Signal(str)
    vmax_ready = Signal(str)
    mask_item_ready = Signal(object)  # QListWidgetItem
    horz_item_ready = Signal(object)  # QListWidgetItem
    shape_ready = Signal(int, int, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._nx = self._ny = self._nz = None
        self._data_loaded = False
        self._load_type = 'base'
        self._transpose = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Dimensions
        dim_sec = CollapsibleSection("Dimensions", collapsed=False)
        dim_row = QHBoxLayout()
        for lbl, attr in [("nx", "nx_edit"), ("ny", "ny_edit"), ("nz", "nz_edit")]:
            dim_row.addWidget(QLabel(lbl))
            edit = LineEdit()
            edit.setValidator(QRegularExpressionValidator(INT_RE, self))
            setattr(self, attr, edit)
            dim_row.addWidget(edit)
        dim_sec.add_layout(dim_row)

        trans_row = QHBoxLayout()
        trans_row.addWidget(QLabel("Transpose"))
        self.trans_radio = RadioGroup(['off', 'on'])
        trans_row.addWidget(self.trans_radio)
        dim_sec.add_layout(trans_row)
        layout.addWidget(dim_sec)

        # Folder loader
        folder_sec = CollapsibleSection("Folder Browse", collapsed=True)
        self.folder_widget = LoadFolderWidget()
        folder_sec.add(self.folder_widget)
        layout.addWidget(folder_sec)

        # Load type + buttons
        load_sec = CollapsibleSection("Load", collapsed=False)
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type"))
        self.load_type_radio = RadioGroup(['base', 'mask', 'horz'])
        type_row.addWidget(self.load_type_radio)
        load_sec.add_layout(type_row)

        btn_row = QHBoxLayout()
        self.load_btn = QPushButton("Load File")
        self.clear_btn = QPushButton("Clear")
        btn_row.addWidget(self.load_btn)
        btn_row.addWidget(self.clear_btn)
        load_sec.add_layout(btn_row)
        layout.addWidget(load_sec)

        self._connect()

    def _connect(self) -> None:
        self.nx_edit.editingFinished.connect(lambda: self._set_dim('nx', self.nx_edit.text()))
        self.ny_edit.editingFinished.connect(lambda: self._set_dim('ny', self.ny_edit.text()))
        self.nz_edit.editingFinished.connect(lambda: self._set_dim('nz', self.nz_edit.text()))
        self.trans_radio.selection_changed.connect(lambda t: setattr(self, '_transpose', t == 'on'))
        self.load_type_radio.selection_changed.connect(lambda t: setattr(self, '_load_type', t))
        self.load_btn.clicked.connect(self.load_file)
        self.folder_widget.current_path.connect(lambda p: self.load_file(p, check=False))

    def _set_dim(self, name: str, text: str) -> None:
        if text:
            setattr(self, f'_{name}', int(text))

    def _get_shape(self):
        if self._nx and self._ny and self._nz:
            return self._nx, self._ny, self._nz
        return None

    def load_file(self, file_path: str = "", check: bool = True) -> None:
        if check:
            if self._data_loaded and self._load_type == 'base':
                QMessageBox.critical(self, "Error", "Data already loaded. Clear first.")
                return
            if not self._data_loaded and self._load_type != 'base':
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

        if self._transpose and data.ndim == 3:
            data = data.T

        if self._load_type == 'base':
            self._data_loaded = True
            nx, ny, nz = data.shape
            self._nx, self._ny, self._nz = nx, ny, nz
            self.nx_edit.set_value(str(nx))
            self.ny_edit.set_value(str(ny))
            self.nz_edit.set_value(str(nz))
            self.shape_ready.emit(nx, ny, nz)
            from cigvis.utils import utils
            v1, v2 = utils.auto_clim(data)
            self.vmin_ready.emit(_format_float(v1))
            self.vmax_ready.emit(_format_float(v2))
            self.base_loaded.emit(data)

        elif self._load_type == 'mask':
            item = QListWidgetItem(Path(file_path).name)
            pw = MaskParamsWidget(mode='3d')
            from cigvis.utils import utils
            v1, v2 = utils.auto_clim(data)
            pw.vmin_edit.set_value(_format_float(v1))
            pw.vmax_edit.set_value(_format_float(v2))
            item.params_widget = pw
            self.mask_item_ready.emit(item)
            self.mask_loaded.emit(data)

        elif self._load_type == 'horz':
            from cigvis.gui.widgets.common import _format_float
            item = QListWidgetItem(Path(file_path).name)
            item.params_widget = HorizonParamsWidget()
            item.params_widget.set_callback = lambda cb: cb
            self.horz_item_ready.emit(item)
            self.horz_loaded.emit(data)

    def _read_file(self, path: str) -> Optional[np.ndarray]:
        # Try to auto-detect dimensions from filename or file type
        if path.endswith('.npy'):
            d = np.load(path)
        elif path.endswith('.vds'):
            import cigvis
            d = cigvis.io.VDSReader(path)
        elif path.endswith(('.sgy', '.segy', '.Sgy')):
            from cigsegy import SegyNP
            d = SegyNP(path)
        else:
            shape = self._get_shape()
            if shape is None:
                # Try to infer from filename
                shape = _get_dim_from_filename(path)
                if not shape:
                    QMessageBox.critical(self, "Error",
                                         "Set nx, ny, nz or use a file with shape in name.")
                    return None
                self._nx, self._ny, self._nz = shape
                self.nx_edit.set_value(str(shape[0]))
                self.ny_edit.set_value(str(shape[1]))
                self.nz_edit.set_value(str(shape[2]))
            nx, ny, nz = self._nx, self._ny, self._nz
            total = nx * ny * nz
            if total * 4 > 1024 ** 3:
                d = np.memmap(path, np.float32, 'c', shape=(nx, ny, nz))
            else:
                d = np.fromfile(path, np.float32).reshape(nx, ny, nz)

        if not hasattr(d, 'shape') or len(d.shape) not in (2, 3):
            QMessageBox.critical(self, "Error", f"Unexpected data shape: {getattr(d, 'shape', '?')}")
            return None

        if self._load_type == 'base' and d.ndim == 2:
            QMessageBox.critical(self, "Error", "Expected 3D data for base volume.")
            return None

        return d

    def set_shape(self, nx: int, ny: int, nz: int) -> None:
        self._nx, self._ny, self._nz = nx, ny, nz
        self.nx_edit.set_value(str(nx))
        self.ny_edit.set_value(str(ny))
        self.nz_edit.set_value(str(nz))

    def sent_data(self, data: np.ndarray) -> None:
        """Programmatically inject pre-loaded data."""
        assert data.ndim == 3
        nx, ny, nz = data.shape
        self.set_shape(nx, ny, nz)
        self._data_loaded = True
        self.shape_ready.emit(nx, ny, nz)
        from cigvis.utils import utils
        v1, v2 = utils.auto_clim(data)
        self.vmin_ready.emit(_format_float(v1))
        self.vmax_ready.emit(_format_float(v2))
        self.base_loaded.emit(data)

    def clear(self, clear_dims: bool = True) -> None:
        self._data_loaded = False
        self._load_type = 'base'
        self.load_type_radio.set_selection('base')
        self.folder_widget.clear()
        if clear_dims:
            self.nx_edit.clear()
            self.ny_edit.clear()
            self.nz_edit.clear()
            self._nx = self._ny = self._nz = None


# ---------------------------------------------------------------------------
# Display panel
# ---------------------------------------------------------------------------

class DisplayPanel3D(QWidget):
    cmap_changed = Signal(str)
    vmin_changed = Signal(str)
    vmax_changed = Signal(str)
    interp_changed = Signal(str)

    def __init__(self, compact: bool = False, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        sec = CollapsibleSection("Base Volume", collapsed=False)
        self.params = ImageParamsWidget(mode='3d', compact=compact)
        sec.add(self.params)
        layout.addWidget(sec)

        self.params.cmap_changed.connect(self.cmap_changed)
        self.params.vmin_changed.connect(self.vmin_changed)
        self.params.vmax_changed.connect(self.vmax_changed)
        self.params.interp_changed.connect(self.interp_changed)

    def set_vmin(self, v: str) -> None:
        self.params.set_vmin(v)

    def set_vmax(self, v: str) -> None:
        self.params.set_vmax(v)

    def set_params(
        self,
        *,
        cmap: Optional[str] = None,
        clim: Optional[Union[list, tuple]] = None,
        interpolation: Optional[str] = None,
    ) -> None:
        vmin = vmax = None
        if clim is not None and len(clim) == 2:
            vmin = _format_float(float(clim[0]))
            vmax = _format_float(float(clim[1]))
        self.params.set_params(
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
        )

    def clear(self) -> None:
        self.params.clear()


# ---------------------------------------------------------------------------
# Camera panel
# ---------------------------------------------------------------------------

class CameraPanel(QWidget):
    azimuth_changed = Signal(int)
    elevation_changed = Signal(int)
    fov_changed = Signal(int)
    aspectx_changed = Signal(float)
    aspecty_changed = Signal(float)
    aspectz_changed = Signal(float)
    update_requested = Signal()
    live_sync_toggled = Signal(bool)

    def __init__(self, compact: bool = False, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        cam_sec = CollapsibleSection("Camera", collapsed=False)
        self.az = SpinBox(); self.az.setRange(0, 360); self.az.setValue(50)
        self.el = SpinBox(); self.el.setRange(-90, 90); self.el.setValue(50)
        self.fov = SpinBox(); self.fov.setRange(1, 179); self.fov.setValue(30)
        if compact:
            for label, spin in [
                ("Azimuth", self.az),
                ("Elevation", self.el),
                ("FOV", self.fov),
            ]:
                row = QHBoxLayout()
                row.addWidget(QLabel(label))
                row.addWidget(spin, 1)
                cam_sec.add_layout(row)
        else:
            r1 = QHBoxLayout()
            r1.addWidget(QLabel("Az"))
            r1.addWidget(self.az)
            r1.addWidget(QLabel("El"))
            r1.addWidget(self.el)
            r1.addWidget(QLabel("FOV"))
            r1.addWidget(self.fov)
            cam_sec.add_layout(r1)

        sync_row = QHBoxLayout()
        self.update_btn = QPushButton("Sync Now")
        self.live_sync_btn = ToggleButton("Live Off", exclusive=False)
        sync_row.addWidget(self.update_btn)
        sync_row.addWidget(self.live_sync_btn)
        cam_sec.add_layout(sync_row)
        layout.addWidget(cam_sec)

        asp_sec = CollapsibleSection("Aspect Ratio", collapsed=False)
        r2 = QVBoxLayout() if compact else QHBoxLayout()
        for lbl, attr in [("x", "aspx"), ("y", "aspy"), ("z", "aspz")]:
            spin = DoubleSpinBox()
            spin.setRange(0.01, 100.0)
            spin.setSingleStep(0.1)
            spin.setValue(1.0)
            setattr(self, attr, spin)
            if compact:
                row = QHBoxLayout()
                row.addWidget(QLabel(lbl))
                row.addWidget(spin, 1)
                r2.addLayout(row)
            else:
                r2.addWidget(QLabel(lbl))
                r2.addWidget(spin)
        asp_sec.add_layout(r2)
        layout.addWidget(asp_sec)

        self.az.changed.connect(self.azimuth_changed)
        self.el.changed.connect(self.elevation_changed)
        self.fov.changed.connect(self.fov_changed)
        self.aspx.changed.connect(self.aspectx_changed)
        self.aspy.changed.connect(self.aspecty_changed)
        self.aspz.changed.connect(self.aspectz_changed)
        self.update_btn.clicked.connect(self.update_requested)
        self.live_sync_btn.clicked.connect(self._on_live_sync_clicked)

    def _on_live_sync_clicked(self, checked: bool) -> None:
        self.live_sync_btn.setText("Live On" if checked else "Live Off")
        self.live_sync_toggled.emit(checked)

    def update_from_params(self, params: list) -> None:
        if not params:
            return
        self.az.blockSignals(True)
        self.el.blockSignals(True)
        self.fov.blockSignals(True)
        self.az.setValue(int(params[0]))
        self.el.setValue(int(params[1]))
        self.fov.setValue(int(params[2]))
        self.az.blockSignals(False)
        self.el.blockSignals(False)
        self.fov.blockSignals(False)

    def clear(self) -> None:
        self.az.setValue(50)
        self.el.setValue(50)
        self.fov.setValue(30)
        self.aspx.setValue(1.0)
        self.aspy.setValue(1.0)
        self.aspz.setValue(1.0)
        self.live_sync_btn.setChecked(False)
        self.live_sync_btn.setText("Live Off")


# ---------------------------------------------------------------------------
# Slices position panel
# ---------------------------------------------------------------------------

class SlicesPanel(QWidget):
    xpos_changed = Signal(int)
    ypos_changed = Signal(int)
    zpos_changed = Signal(int)

    def __init__(self, compact: bool = False, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        sec = CollapsibleSection("Slice Positions", collapsed=False)
        row = QVBoxLayout() if compact else QHBoxLayout()
        for lbl, attr, sig in [("x", "xpos", "xpos_changed"),
                                ("y", "ypos", "ypos_changed"),
                                ("z", "zpos", "zpos_changed")]:
            spin = SpinBox()
            spin.setMinimum(0)
            setattr(self, attr, spin)
            if compact:
                subrow = QHBoxLayout()
                subrow.addWidget(QLabel(lbl))
                subrow.addWidget(spin, 1)
                row.addLayout(subrow)
            else:
                row.addWidget(QLabel(lbl))
                row.addWidget(spin)
            getattr(spin, 'changed').connect(getattr(self, sig))
        sec.add_layout(row)
        layout.addWidget(sec)

    def set_limits(self, nx: int, ny: int, nz: int) -> None:
        self.set_axis_limits({
            'x': (0, nx - 1),
            'y': (0, ny - 1),
            'z': (0, nz - 1),
        })

    def set_axis_limits(self, limits: dict) -> None:
        for axis, spin in [('x', self.xpos), ('y', self.ypos), ('z', self.zpos)]:
            lo, hi = limits.get(axis, (0, 0))
            spin.setRange(int(lo), max(int(hi), int(lo)))

    def set_positions(self, positions: dict) -> None:
        for axis, spin in [('x', self.xpos), ('y', self.ypos), ('z', self.zpos)]:
            if axis in positions:
                spin.blockSignals(True)
                spin.setValue(int(positions[axis]))
                spin.blockSignals(False)

    def clear(self) -> None:
        self.xpos.setValue(0)
        self.ypos.setValue(0)
        self.zpos.setValue(0)


# ---------------------------------------------------------------------------
# Horizon params widget (for HorizonMixin)
# ---------------------------------------------------------------------------

class HorizonParamsWidget(QWidget):
    """Per-horizon parameter controls."""

    def __init__(self, update_callback: Optional[Callable] = None, parent=None) -> None:
        super().__init__(parent)
        self._cb = update_callback
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        off_row = QHBoxLayout()
        off_row.addWidget(QLabel("Offset"))
        self.ox = LineEdit(); self.ox.setText('0')
        self.oy = LineEdit(); self.oy.setText('0')
        self.oz = LineEdit(); self.oz.setText('0')
        for w in [self.ox, self.oy, self.oz]:
            off_row.addWidget(w)
        layout.addLayout(off_row)

        itv_row = QHBoxLayout()
        itv_row.addWidget(QLabel("Interval"))
        self.ix = LineEdit(); self.ix.setText('1')
        self.iy = LineEdit(); self.iy.setText('1')
        self.iz = LineEdit(); self.iz.setText('1')
        for w in [self.ix, self.iy, self.iz]:
            itv_row.addWidget(w)
        layout.addLayout(itv_row)

        ctrl_row = QHBoxLayout()
        self.update_btn = QPushButton("update")
        val_lbl = QLabel("value")
        self.value_combo = EditableComboBox()
        self.value_combo.addItems(['depth', 'amplitude'])
        self.value_combo.setCurrentText('depth')
        cmap_lbl = QLabel("cmap")
        self.cmap_combo = EditableComboBox()
        self.cmap_combo.addItems(['jet', 'gray', 'seismic', 'Petrel', 'stratum'])
        self.cmap_combo.setCurrentText('jet')
        ctrl_row.addWidget(self.update_btn)
        ctrl_row.addWidget(val_lbl)
        ctrl_row.addWidget(self.value_combo)
        ctrl_row.addWidget(cmap_lbl)
        ctrl_row.addWidget(self.cmap_combo)
        layout.addLayout(ctrl_row)

        self.update_btn.clicked.connect(self._on_update)
        self.value_combo.changed.connect(self._on_value)
        self.cmap_combo.changed.connect(self._on_cmap)

    def set_callback(self, cb: Callable) -> None:
        self._cb = cb
        self._on_update(init=True)
        self._on_value(self.value_combo.currentText(), init=True)
        self._on_cmap(self.cmap_combo.currentText(), init=True)

    def _on_update(self, init: bool = False) -> None:
        if self._cb:
            idx = -1 if init else None
            offset = [float(self.ox.text() or '0'),
                      float(self.oy.text() or '0'),
                      float(self.oz.text() or '0')]
            itv = [float(self.ix.text() or '1'),
                   float(self.iy.text() or '1'),
                   float(self.iz.text() or '1')]
            self._cb('coord', [offset, itv], idx)

    def _on_value(self, text: str, init: bool = False) -> None:
        if self._cb:
            idx = -1 if init else None
            t = 'amp' if text == 'amplitude' else text
            self._cb('value_type', t, idx)

    def _on_cmap(self, text: str, init: bool = False) -> None:
        if self._cb:
            idx = -1 if init else None
            self._cb('cmap', text, idx)


# ---------------------------------------------------------------------------
# Overlays panel (masks + horizons)
# ---------------------------------------------------------------------------

class OverlaysPanel3D(QWidget):
    mask_params_changed = Signal(list)
    mask_deleted = Signal(int)
    horz_params_changed = Signal(list)
    horz_deleted = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Masks
        mask_sec = CollapsibleSection("Masks", collapsed=False)
        self.mask_items = ItemsWidget()
        self.mask_items.params_changed.connect(self.mask_params_changed)
        self.mask_items.item_deleted.connect(self.mask_deleted)
        mask_sec.add(self.mask_items)
        layout.addWidget(mask_sec)

        # Horizons
        horz_sec = CollapsibleSection("Horizons", collapsed=False)

        self.horz_items = ItemsWidget()
        self.horz_items.params_changed.connect(self.horz_params_changed)
        self.horz_items.item_deleted.connect(self.horz_deleted)
        horz_sec.add(self.horz_items)
        layout.addWidget(horz_sec)

    def add_mask_item(self, item: QListWidgetItem) -> None:
        self.mask_items.add_item(item)

    def set_mask_items(self, params_list: List[dict]) -> None:
        self.mask_items.clear()
        for idx, params in enumerate(params_list):
            self.add_mask_item(self._make_mask_item(idx, params))

    def _make_mask_item(self, idx: int, params: dict) -> QListWidgetItem:
        item = QListWidgetItem(str(params.get('name') or f'mask_{idx}'))
        pw = MaskParamsWidget(mode='3d')
        clim = params.get('clim')
        vmin = params.get('vmin')
        vmax = params.get('vmax')
        if clim is not None and len(clim) == 2:
            vmin, vmax = clim
        pw.set_params(
            cmap=params.get('cmap'),
            vmin=_format_float(float(vmin)) if vmin is not None else None,
            vmax=_format_float(float(vmax)) if vmax is not None else None,
            interpolation=params.get('interpolation'),
            alpha=params.get('alpha'),
            excpt=params.get('except'),
        )
        item.params_widget = pw
        return item

    def add_horz_item(self, item: QListWidgetItem) -> None:
        self.horz_items.add_item(item)

    def clear(self) -> None:
        self.mask_items.clear()
        self.horz_items.clear()


# ---------------------------------------------------------------------------
# SAM panel (optional interactive segmentation)
# ---------------------------------------------------------------------------

class SamPanel(QWidget):
    """
    SAM-like interactive segmentation panel.

    Activate SAM mode and the canvas will:
      - Alt+LeftClick → add prompt point
      - Enter         → run inference (async)
      - Delete        → undo last prompt
      - C             → clear all prompts

    Set `decode_fn` before activating for actual inference.
    """

    sam_toggled = Signal(bool)
    run_requested = Signal()
    clear_requested = Signal()
    undo_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        sec = CollapsibleSection("SAM Segmentation", collapsed=False)

        # Enable toggle
        en_row = QHBoxLayout()
        en_row.addWidget(QLabel("SAM Mode"))
        self.enable_btn = ToggleButton("Off")
        en_row.addWidget(self.enable_btn)
        sec.add_layout(en_row)

        # Prompt count display
        self.prompt_label = QLabel("Prompts: 0")
        sec.add(self.prompt_label)

        # Buttons
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run ⏎")
        self.run_btn.setObjectName("AccentButton")
        self.undo_btn = QPushButton("Undo")
        self.clear_btn = QPushButton("Clear")
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.undo_btn)
        btn_row.addWidget(self.clear_btn)
        sec.add_layout(btn_row)

        # Hint
        hint = QLabel("Alt+Click to add prompts,\nEnter to run inference.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: rgba(255,255,255,0.45); font-size: 10px;")
        sec.add(hint)

        layout.addWidget(sec)
        self.setEnabled(False)

        self.enable_btn.clicked.connect(self._on_toggle)
        self.run_btn.clicked.connect(self.run_requested)
        self.undo_btn.clicked.connect(self.undo_requested)
        self.clear_btn.clicked.connect(self.clear_requested)

    def _on_toggle(self, checked: bool) -> None:
        self.enable_btn.setText("On" if checked else "Off")
        self.sam_toggled.emit(checked)

    def set_prompt_count(self, n: int) -> None:
        self.prompt_label.setText(f"Prompts: {n}")

    def enable_sam(self) -> None:
        self.setEnabled(True)


# ---------------------------------------------------------------------------
# Filename → dimension helper
# ---------------------------------------------------------------------------

def _get_dim_from_filename(fname: str):
    """Try to infer (nx, ny, nz) from filename patterns."""
    import cigvis
    stem = Path(fname).stem
    p1 = re.search(r'\w+\_h(\d+)x(\d+)x(\d+)', stem)
    if p1:
        z, y, x = p1.groups()
        return int(x), int(y), int(z)
    p2 = re.search(r'\w+\_(\d+)\_(\d+)\_(\d+)', stem)
    if p2:
        x, y, z = p2.groups()
        return int(x), int(y), int(z)
    return None
