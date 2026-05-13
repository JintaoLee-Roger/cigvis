"""Common PySide6 custom widgets for CigVis GUI."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Callable

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit,
    QSpinBox, QDoubleSpinBox, QComboBox, QPushButton,
    QRadioButton, QButtonGroup, QListWidget, QListWidgetItem,
    QMenu, QFileDialog, QMessageBox, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QRegularExpression
from PySide6.QtGui import QRegularExpressionValidator

INT_RE = QRegularExpression(r"^[1-9][0-9]*$")
FLOAT_RE = QRegularExpression(r"[-+]?[0-9]*\.?[0-9]+")


def _sortfile(flist: List[Path]) -> List[Path]:
    try:
        return sorted(flist, key=lambda x: int(x.stem))
    except Exception:
        return sorted(flist)


# ---------------------------------------------------------------------------
# Basic input widgets
# ---------------------------------------------------------------------------

class LineEdit(QLineEdit):
    """QLineEdit that can set text and emit editingFinished atomically."""

    def set_value(self, text: str) -> None:
        self.setText(text)
        self.editingFinished.emit()


class SpinBox(QSpinBox):
    """QSpinBox that emits `changed` on step and Enter."""

    changed = Signal(int)

    def stepBy(self, steps: int) -> None:
        super().stepBy(steps)
        self.changed.emit(self.value())

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.changed.emit(self.value())
        elif event.key() == Qt.Key_Up:
            if self.value() < self.maximum():
                v = self.value() + 1
                self.setValue(v)
                self.changed.emit(v)
        elif event.key() == Qt.Key_Down:
            if self.value() > self.minimum():
                v = self.value() - 1
                self.setValue(v)
                self.changed.emit(v)
        else:
            super().keyPressEvent(event)


class DoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that emits `changed` on step and Enter."""

    changed = Signal(float)

    def stepBy(self, steps: int) -> None:
        super().stepBy(steps)
        self.changed.emit(self.value())

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.changed.emit(self.value())
        else:
            super().keyPressEvent(event)


class EditableComboBox(QComboBox):
    """QComboBox that emits `changed` on selection or Enter."""

    changed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.NoInsert)
        self.activated.connect(lambda: self.changed.emit(self.currentText()))

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.changed.emit(self.currentText())
        else:
            super().keyPressEvent(event)


class ToggleButton(QPushButton):
    """Checkable push button that optionally manages mutual exclusion via parent."""

    def __init__(self, title: str = "", exclusive: bool = True, parent=None) -> None:
        super().__init__(title, parent)
        self.setCheckable(True)
        self._exclusive = exclusive

    def nextCheckState(self) -> None:
        super().nextCheckState()
        if self._exclusive and self.isChecked():
            parent = self.parent()
            if parent and hasattr(parent, 'update_toggle_state'):
                parent.update_toggle_state(self, True)


class RadioGroup(QWidget):
    """Horizontal (or vertical) group of radio buttons."""

    selection_changed = Signal(str)

    def __init__(self, names: List[str], horizontal: bool = True, parent=None) -> None:
        super().__init__(parent)
        self._names = names
        self._btns: List[QRadioButton] = []

        layout = QHBoxLayout(self) if horizontal else QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._group = QButtonGroup(self)
        for i, name in enumerate(names):
            btn = QRadioButton(name)
            layout.addWidget(btn)
            self._btns.append(btn)
            self._group.addButton(btn, i)
            btn.toggled.connect(self._on_toggled)

        if self._btns:
            self._btns[0].setChecked(True)
        self.selected = names[0] if names else ""

    def _on_toggled(self) -> None:
        btn = self._group.checkedButton()
        if btn is not None:
            self.selected = btn.text()
            self.selection_changed.emit(self.selected)

    def set_selection(self, name: str) -> None:
        for btn in self._btns:
            if btn.text() == name:
                btn.setChecked(True)
                break

    def get_selection(self) -> str:
        return self.selected


class RectP:
    """Rectangle defined by two corner points (used for box annotation)."""

    def __init__(self, x0=None, y0=None, x1=None, y1=None) -> None:
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1

    def add_p0(self, x0, y0) -> None:
        self.x0, self.y0 = x0, y0

    def add_p1(self, x1, y1) -> None:
        self.x1, self.y1 = x1, y1

    def to_points(self) -> list:
        return [self.x0, self.y0, self.x1, self.y1]

    def to_start_size(self) -> list:
        return [self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0)]


# ---------------------------------------------------------------------------
# Parameter sub-widgets
# ---------------------------------------------------------------------------

class ImageParamsWidget(QWidget):
    """Colormap / clim / interpolation controls."""

    cmap_changed = Signal(str)
    vmin_changed = Signal(str)
    vmax_changed = Signal(str)
    interp_changed = Signal(str)

    CMAPS = [
        'gray', 'seismic', 'Petrel', 'stratum', 'jet',
        'od_seismic1', 'bwp', 'od_seismic2', 'od_seismic3',
    ]
    INTERPS_2D = [
        'none', 'nearest', 'bilinear', 'bicubic', 'antialiased',
        'spline36', 'hamming', 'gaussian', 'lanczos',
    ]
    INTERPS_3D = [
        'nearest', 'linear', 'bicubic', 'bilinear', 'cubic', 'sinc',
        'blackman', 'catrom', 'bessel', 'gaussian', 'hamming',
        'hanning', 'hermite', 'kaiser', 'lanczos', 'mitchell',
        'quadric', 'spline16', 'spline36',
    ]

    def __init__(self, mode: str = '3d', compact: bool = False,
                 parent=None) -> None:
        super().__init__(parent)
        self._params_cb = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5 if compact else 3)

        def add_field(label: str, widget: QWidget) -> None:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(QLabel(label))
            row.addWidget(widget, 1)
            layout.addLayout(row)

        # clim row
        self.vmin_edit = LineEdit()
        self.vmin_edit.setValidator(QRegularExpressionValidator(FLOAT_RE, self))
        self.vmax_edit = LineEdit()
        self.vmax_edit.setValidator(QRegularExpressionValidator(FLOAT_RE, self))

        # cmap + interp row
        self.cmap_combo = EditableComboBox()
        self.cmap_combo.addItems(self.CMAPS)
        self.cmap_combo.setCurrentText('gray')

        self.interp_combo = QComboBox()
        interps = self.INTERPS_2D if mode == '2d' else self.INTERPS_3D
        self.interp_combo.addItems(interps)
        self.interp_combo.setCurrentText('bilinear')

        if compact:
            add_field("vmin", self.vmin_edit)
            add_field("vmax", self.vmax_edit)
            add_field("cmap", self.cmap_combo)
            add_field("interp", self.interp_combo)
        else:
            clim_row = QHBoxLayout()
            clim_row.addWidget(QLabel("vmin"))
            clim_row.addWidget(self.vmin_edit)
            clim_row.addWidget(QLabel("vmax"))
            clim_row.addWidget(self.vmax_edit)
            layout.addLayout(clim_row)

            style_row = QHBoxLayout()
            style_row.addWidget(QLabel("cmap"))
            style_row.addWidget(self.cmap_combo)
            layout.addLayout(style_row)

            interp_row = QHBoxLayout()
            interp_row.addWidget(QLabel("interp"))
            interp_row.addWidget(self.interp_combo)
            layout.addLayout(interp_row)

        # connections
        self.cmap_combo.changed.connect(self.cmap_changed)
        self.vmin_edit.editingFinished.connect(
            lambda: self.vmin_changed.emit(self.vmin_edit.text()))
        self.vmax_edit.editingFinished.connect(
            lambda: self.vmax_changed.emit(self.vmax_edit.text()))
        self.interp_combo.currentTextChanged.connect(self.interp_changed)
        self.cmap_changed.connect(lambda value: self._emit_param('cmap', value))
        self.vmin_changed.connect(lambda value: self._emit_param('vmin', value))
        self.vmax_changed.connect(lambda value: self._emit_param('vmax', value))
        self.interp_changed.connect(lambda value: self._emit_param('interp', value))

    def set_callback(self, cb: Callable) -> None:
        self._params_cb = cb

    def _emit_param(self, mode: str, value) -> None:
        if self._params_cb is not None:
            self._params_cb(mode, value)

    def set_vmin(self, v: str) -> None:
        self.vmin_edit.set_value(v)

    def set_vmax(self, v: str) -> None:
        self.vmax_edit.set_value(v)

    def set_params(
        self,
        *,
        cmap: Optional[str] = None,
        vmin: Optional[str] = None,
        vmax: Optional[str] = None,
        interpolation: Optional[str] = None,
    ) -> None:
        widgets = [
            self.vmin_edit,
            self.vmax_edit,
            self.cmap_combo,
            self.interp_combo,
        ]
        for widget in widgets:
            widget.blockSignals(True)
        try:
            if vmin is not None:
                self.vmin_edit.setText(vmin)
            if vmax is not None:
                self.vmax_edit.setText(vmax)
            if cmap:
                self.cmap_combo.setCurrentText(cmap)
            if interpolation:
                self.interp_combo.setCurrentText(interpolation)
        finally:
            for widget in widgets:
                widget.blockSignals(False)

    def clear(self) -> None:
        self.vmin_edit.clear()
        self.vmax_edit.clear()
        self.cmap_combo.setCurrentText('gray')
        self.interp_combo.setCurrentText('bilinear')


class MaskParamsWidget(ImageParamsWidget):
    """Mask overlay parameters: adds alpha and except controls."""

    alpha_changed = Signal(float)
    except_changed = Signal(str)

    EXCEPTS = ['None', 'min', 'ramp', 'max']

    def __init__(self, mode: str = '3d', parent=None) -> None:
        super().__init__(mode=mode, parent=parent)
        self.cmap_combo.setCurrentText('jet')
        self.interp_combo.setCurrentText('nearest')

        extra_row = QHBoxLayout()
        extra_row.addWidget(QLabel("alpha"))
        self.alpha_spin = DoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(0.5)
        extra_row.addWidget(self.alpha_spin)

        extra_row.addWidget(QLabel("except"))
        self.except_combo = EditableComboBox()
        self.except_combo.addItems(self.EXCEPTS)
        self.except_combo.setCurrentText('None')
        extra_row.addWidget(self.except_combo)

        self.layout().addLayout(extra_row)

        self.alpha_spin.changed.connect(self.alpha_changed)
        self.except_combo.changed.connect(self.except_changed)
        self.alpha_changed.connect(lambda value: self._emit_param('alpha', value))
        self.except_changed.connect(lambda value: self._emit_param('except', value))

    def set_params(
        self,
        *,
        cmap: Optional[str] = None,
        vmin: Optional[str] = None,
        vmax: Optional[str] = None,
        interpolation: Optional[str] = None,
        alpha: Optional[float] = None,
        excpt: Optional[str] = None,
    ) -> None:
        super().set_params(
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
        )
        widgets = [self.alpha_spin, self.except_combo]
        for widget in widgets:
            widget.blockSignals(True)
        try:
            if alpha is not None:
                self.alpha_spin.setValue(float(alpha))
            if excpt:
                self.except_combo.setCurrentText(excpt)
        finally:
            for widget in widgets:
                widget.blockSignals(False)


# ---------------------------------------------------------------------------
# List widget for layers (masks, horizons)
# ---------------------------------------------------------------------------

class ItemsWidget(QWidget):
    """
    List of named items (masks / horizons) with per-item params popups.

    Signals
    -------
    params_changed : [idx, mode, value]
    item_deleted   : idx
    """

    params_changed = Signal(list)
    item_deleted = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)

        self.list_widget = QListWidget()
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._context_menu)
        self.list_widget.itemClicked.connect(self._show_params)
        self._layout.addWidget(self.list_widget, 1)

        self._current_params: Optional[QWidget] = None

    def add_item(self, item: QListWidgetItem) -> None:
        self.list_widget.addItem(item)
        item.params_widget.set_callback(self._update_callback)

    def _show_params(self, item: QListWidgetItem) -> None:
        if self._current_params is not None:
            self._layout.removeWidget(self._current_params)
            self._current_params.hide()
        self._current_params = item.params_widget
        self._layout.addWidget(self._current_params, 2)
        self._current_params.show()

    def _context_menu(self, pos) -> None:
        menu = QMenu(self)
        delete_action = menu.addAction("Delete")
        action = menu.exec(self.list_widget.mapToGlobal(pos))
        if action == delete_action:
            self._remove_selected()

    def _remove_selected(self) -> None:
        item = self.list_widget.currentItem()
        idx = self.list_widget.currentRow()
        if item is None:
            return
        pw = item.params_widget
        if pw == self._current_params:
            self._layout.removeWidget(pw)
            pw.setParent(None)
            self._current_params = None
        self.list_widget.takeItem(idx)
        del item
        self.item_deleted.emit(idx)

    def _update_callback(self, mode: str, value, idx=None) -> None:
        if idx is None:
            idx = self.list_widget.currentRow()
        self.params_changed.emit([idx, mode, value])

    def clear(self) -> None:
        while self.list_widget.count() > 0:
            item = self.list_widget.takeItem(0)
            if hasattr(item, 'params_widget') and item.params_widget:
                item.params_widget.setParent(None)
            del item


# ---------------------------------------------------------------------------
# Folder-based data loader
# ---------------------------------------------------------------------------

class LoadFolderWidget(QWidget):
    """Browse folder of .dat/.npy files; emits current file path on index change."""

    current_path = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._flist: List[Path] = []

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        self._btn = QPushButton("Folder")
        self._btn.setFixedWidth(52)
        row.addWidget(self._btn)

        self._idx_spin = SpinBox()
        self._idx_spin.setFixedWidth(52)
        row.addWidget(self._idx_spin)

        self._name_edit = QLineEdit("None")
        self._name_edit.setReadOnly(True)
        row.addWidget(self._name_edit)

        self._btn.clicked.connect(self._load_folder)
        self._idx_spin.changed.connect(self._on_idx_changed)

    def _load_folder(self, path: str = "") -> None:
        if not path:
            path = QFileDialog.getExistingDirectory(self, "Select Folder", "")
        if not path:
            return

        flist = list(Path(path).glob('*.dat'))
        flist += list(Path(path).glob('*.npy'))
        flist = [f for f in flist if not f.name.startswith('.')]
        flist = _sortfile(flist)
        if not flist:
            QMessageBox.warning(self, "Empty", "No .dat or .npy files found.")
            return

        self._flist = flist
        self._idx_spin.setRange(0, len(flist) - 1)
        self._idx_spin.setValue(0)
        self._on_idx_changed(0)

    def _on_idx_changed(self, idx: int) -> None:
        if not self._flist:
            return
        idx = int(idx)
        self._name_edit.setText(self._flist[idx].name)
        self.current_path.emit(str(self._flist[idx]))

    def clear(self) -> None:
        self._flist = []
        self._idx_spin.setValue(0)
        self._name_edit.setText("None")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_float(f: float) -> str:
    return f'{f:.2f}' if abs(f) >= 1 else f'{f:.2g}'
