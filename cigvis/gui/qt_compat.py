"""Small Qt binding compatibility layer for the CigVis GUI."""

from __future__ import annotations

import importlib
import os
from typing import Iterable, Optional


_BINDINGS = {
    "pyside6": ("PySide6", "Signal", "Slot"),
    "pyqt6": ("PyQt6", "pyqtSignal", "pyqtSlot"),
    "pyqt5": ("PyQt5", "pyqtSignal", "pyqtSlot"),
}
_DEFAULT_ORDER = ("pyside6", "pyqt6", "pyqt5")


def _requested_api() -> Optional[str]:
    value = os.environ.get("CIGVIS_QT_API") or os.environ.get("QT_API")
    if not value:
        return None
    key = value.lower().replace("-", "").replace("_", "")
    aliases = {
        "pyside6": "pyside6",
        "pyqt6": "pyqt6",
        "pyqt5": "pyqt5",
    }
    return aliases.get(key)


def _candidate_order() -> Iterable[str]:
    requested = _requested_api()
    if requested is not None:
        yield requested
    for name in _DEFAULT_ORDER:
        if name != requested:
            yield name


def _load_binding():
    errors = []
    for api in _candidate_order():
        package, signal_name, slot_name = _BINDINGS[api]
        try:
            qtcore = importlib.import_module(f"{package}.QtCore")
            qtgui = importlib.import_module(f"{package}.QtGui")
            qtwidgets = importlib.import_module(f"{package}.QtWidgets")
        except ImportError as exc:
            errors.append(f"{package}: {exc}")
            continue
        return api, package, qtcore, qtgui, qtwidgets, signal_name, slot_name
    detail = "; ".join(errors) if errors else "no supported Qt binding found"
    raise ImportError(
        "CIGVis GUI requires PySide6, PyQt6, or PyQt5. "
        "PySide6 is the default recommendation: `pip install \"cigvis[gui]\"` "
        "or `pip install PySide6`. "
        f"Import attempts: {detail}"
    )


class _EnumNamespace:
    def __init__(self, namespace, groups):
        self._namespace = namespace
        self._groups = groups

    def __getattr__(self, name):
        if hasattr(self._namespace, name):
            return getattr(self._namespace, name)
        for group_name in self._groups:
            group = getattr(self._namespace, group_name, None)
            if group is not None and hasattr(group, name):
                return getattr(group, name)
        raise AttributeError(name)


def _enum_value(owner, name: str, group_name: str):
    if hasattr(owner, name):
        return getattr(owner, name)
    group = getattr(owner, group_name)
    return getattr(group, name)


QT_API, QT_PACKAGE, QtCore, QtGui, QtWidgets, _SIGNAL, _SLOT = _load_binding()

Qt = _EnumNamespace(
    QtCore.Qt,
    ("Key", "WidgetAttribute", "ContextMenuPolicy", "ScrollBarPolicy", "ShortcutContext"),
)
QEvent = _EnumNamespace(QtCore.QEvent, ("Type",))
QRegularExpression = QtCore.QRegularExpression
Signal = getattr(QtCore, _SIGNAL)
Slot = getattr(QtCore, _SLOT, None)

QPoint = QtCore.QPoint
QPropertyAnimation = QtCore.QPropertyAnimation
QEasingCurve = QtCore.QEasingCurve
QTimer = QtCore.QTimer

QFont = QtGui.QFont
QKeySequence = QtGui.QKeySequence
QRegularExpressionValidator = QtGui.QRegularExpressionValidator
QShortcut = getattr(QtGui, "QShortcut", getattr(QtWidgets, "QShortcut", None))

QApplication = QtWidgets.QApplication
QButtonGroup = QtWidgets.QButtonGroup
QComboBox = QtWidgets.QComboBox
QDoubleSpinBox = QtWidgets.QDoubleSpinBox
QFileDialog = QtWidgets.QFileDialog
QFrame = QtWidgets.QFrame
QHBoxLayout = QtWidgets.QHBoxLayout
QLabel = QtWidgets.QLabel
QLineEdit = QtWidgets.QLineEdit
QListWidget = QtWidgets.QListWidget
QListWidgetItem = QtWidgets.QListWidgetItem
QMainWindow = QtWidgets.QMainWindow
QMenu = QtWidgets.QMenu
QMessageBox = QtWidgets.QMessageBox
QPushButton = QtWidgets.QPushButton
QRadioButton = QtWidgets.QRadioButton
QScrollArea = QtWidgets.QScrollArea
QSizePolicy = QtWidgets.QSizePolicy
QSpinBox = QtWidgets.QSpinBox
QStackedWidget = QtWidgets.QStackedWidget
QStatusBar = QtWidgets.QStatusBar
QToolButton = QtWidgets.QToolButton
QVBoxLayout = QtWidgets.QVBoxLayout
QWidget = QtWidgets.QWidget

ComboBoxNoInsert = _enum_value(QComboBox, "NoInsert", "InsertPolicy")
EasingOutCubic = _enum_value(QEasingCurve, "OutCubic", "Type")
FrameHLine = _enum_value(QFrame, "HLine", "Shape")
SizePolicyExpanding = _enum_value(QSizePolicy, "Expanding", "Policy")
SizePolicyFixed = _enum_value(QSizePolicy, "Fixed", "Policy")
