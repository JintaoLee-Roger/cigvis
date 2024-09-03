# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Some custom widgets
"""
from typing import List
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore, QtGui
import numpy as np
from pathlib import Path

CANVAS_SIZE = (800, 600)  # (width, height)

INT_validator = QtCore.QRegExp(r"^[1-9][0-9]*$")
FLOAT_validator = QtCore.QRegExp(r"[-+]?[0-9]*\.?[0-9]+")


def _sortfile(flist: List[Path]):
    try:
        flist = sorted(flist, key=lambda x: int(x.stem))
    except:
        flist = sorted(flist)

    return flist


class EditableComboBox(qtw.QComboBox):
    changed = QtCore.pyqtSignal(str)  # 自定义的信号

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.activated.connect(self.on_changed)
        self.setEditable(True)  # 允许手动输入
        self.setInsertPolicy(qtw.QComboBox.NoInsert)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return:
            self.changed.emit(self.currentText())  # 当按下Enter键时，发出自定义信号
        else:
            super().keyPressEvent(event)

    def on_changed(self):
        self.changed.emit(self.currentText())


class MyQSpinBox(qtw.QSpinBox):
    changed = QtCore.pyqtSignal(int)

    def stepBy(self, steps):
        super().stepBy(steps)
        self.changed.emit(self.value())

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return:
            self.changed.emit(self.value())
        elif event.key() == QtCore.Qt.Key_Up:
            if self.value() >= self.maximum():
                return
            value = self.value() + 1
            self.setValue(value)
            self.changed.emit(value)
        elif event.key() == QtCore.Qt.Key_Down:
            if self.value() <= self.minimum():
                return
            value = self.value() - 1
            self.setValue(value)
            self.changed.emit(value)
        else:
            super().keyPressEvent(event)


class MyQDoubleSpinBox(qtw.QDoubleSpinBox):
    changed = QtCore.pyqtSignal(float)

    def stepBy(self, steps):
        super().stepBy(steps)
        self.changed.emit(self.value())

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return:
            self.changed.emit(self.value())
        else:
            super().keyPressEvent(event)


class MyQLineEdit(qtw.QLineEdit):

    def setTextAndEmit(self, text):
        self.setText(text)
        # 手动调用处理函数
        self.editingFinished.emit()


class UnpickRadioButton(qtw.QRadioButton):

    def __init__(self, title, parent=None):
        super(UnpickRadioButton, self).__init__(title, parent)

    def mousePressEvent(self, event):
        # 重写鼠标按下事件，但不调用基类的事件处理，阻止选中状态的改变
        pass

    def mouseReleaseEvent(self, event):
        # 重写鼠标释放事件，同样不调用基类的事件处理
        pass


class RadioButtonPanel(qtw.QWidget):
    selectionChanged = QtCore.pyqtSignal(str)

    def __init__(self, names, picked=True, hori=True, parent=None):
        super(RadioButtonPanel, self).__init__(parent)
        self.names = names
        self.radioButtons = []
        radio = qtw.QRadioButton if picked else UnpickRadioButton

        # 选择布局方向
        self.layout = qtw.QHBoxLayout(self) if hori else qtw.QVBoxLayout(self)

        # 创建并添加单选按钮到布局
        for name in names:
            radioButton = radio(name)
            self.layout.addWidget(radioButton)
            self.radioButtons.append(radioButton)
            # 连接单选按钮的信号
            radioButton.toggled.connect(self.onRadioButtonChanged)

        # 设置第一个单选按钮为默认选中状态
        self.radioButtons[0].setChecked(True)
        self.selected = self.names[0]

    def onRadioButtonChanged(self):
        # 当单选按钮的状态改变时触发
        radioButton = self.sender()
        if radioButton.isChecked():
            self.selected = radioButton.text()
            self.selectionChanged.emit(self.selected)
            for rad in self.radioButtons:
                if rad != radioButton:
                    rad.setChecked(False)

    def getCurrentSelection(self):
        return self.selected


class ToggleButton(qtw.QPushButton):

    def __init__(self, title=None, usesfix=False, exclu=True, parent=None):
        if usesfix:
            title = title + " OFF"
        self.usesfix = usesfix
        super().__init__(title, parent)
        self.title = title
        self.exclu = exclu
        self.setCheckable(True)
        self.toggled.connect(self.onToggle)

    def onToggle(self, checked):
        if self.exclu:
            parent = self.parent()
            if parent:
                parent.updateToggleState(self, checked)

        if not self.usesfix:
            return
        if checked:
            self.setText(self.title + " ON")
        else:
            self.setText(self.title + " OFF")


class RectP:

    def __init__(self,
                 x0: float = None,
                 y0: float = None,
                 x1: float = None,
                 y1: float = None) -> None:
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def add_p0(self, x0, y0):
        self.x0 = x0
        self.y0 = y0

    def add_p1(self, x1, y1):
        self.x1 = x1
        self.y1 = y1

    def add_p0event(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata

    def add_p1event(self, event):
        self.x1 = event.xdata
        self.y1 = event.ydata

    @classmethod
    def from_points(cls, p0: List, p1: List):
        # p1, p2可以是任何有x和y属性的对象
        return cls(p0[0], p0[1], p1[0], p1[1])

    @classmethod
    def from_events(cls, e0, e1):
        return cls(e0.xdata, e0.ydata, e1.xdata, e1.ydata)

    def to_start_size(self):
        width = abs(self.x1 - self.x0)
        height = abs(self.y1 - self.y0)
        return [self.x0, self.y0, width, height]

    def to_points(self):
        return [self.x0, self.y0, self.x1, self.y1]


class BaseWidget(qtw.QWidget):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

    def addwidgets(self, layout, widgets):
        for widget in widgets:
            layout.addWidget(widget)

    def addlayout(self, layout, sublayouts):
        for sublayout in sublayouts:
            layout.addLayout(sublayout)


class ImageParams(BaseWidget):

    def __init__(self, interps=None, parent=None) -> None:
        super().__init__(parent)

        self.ilayout = qtw.QVBoxLayout()
        self.ilayout.setContentsMargins(0, 0, 0, 0)
        self.ilayout.setSpacing(0)

        # clim
        clim_layout = qtw.QHBoxLayout()
        vmin_label = qtw.QLabel('vmin:')
        self.vmin_input = MyQLineEdit()
        self.vmin_input.setValidator(
            QtGui.QRegExpValidator(FLOAT_validator, self))
        vmax_label = qtw.QLabel('vmax:')
        self.vmax_input = MyQLineEdit()
        self.vmax_input.setValidator(
            QtGui.QRegExpValidator(FLOAT_validator, self))
        self.addwidgets(
            clim_layout,
            [vmin_label, self.vmin_input, vmax_label, self.vmax_input])

        # colormap
        layout2 = qtw.QHBoxLayout()
        colormap_label = qtw.QLabel('cmap:')
        self.colormap_combo = EditableComboBox()
        colormaps = [
            'gray', 'seismic', 'Petrel', 'stratum', 'jet', 'od_seismic1', 'bwp',
            'od_seismic2', 'od_seismic3'
        ]
        self.colormap_combo.addItems(colormaps)
        self.colormap_combo.setCurrentText('gray')  # 默认值为'gray'

        # interpolation
        interp_label = qtw.QLabel('Interp:')
        self.interp_combo = qtw.QComboBox()
        if interps is None:
            interps = [
                'none', 'nearest', 'bilinear', 'bicubic', 'quadric', 'sinc',
                'blackman', 'antialiased', 'spline36', 'mitchell', 'hamming',
                'catrom', 'gaussian', 'hanning', 'lanczos', 'bessel',
                'spline16', 'kaiser', 'hermite'
            ]
        self.interp_combo.addItems(interps)
        self.interp_combo.setCurrentText('bilinear')

        self.addwidgets(layout2, [
            colormap_label, self.colormap_combo, interp_label,
            self.interp_combo
        ])

        sublayouts = [clim_layout, layout2]
        self.addlayout(self.ilayout, sublayouts)
        self.setLayout(self.ilayout)

    def clear(self):
        self.vmin_input.clear()
        self.vmax_input.clear()
        self.colormap_combo.setCurrentText('gray')
        self.interp_combo.setCurrentText('bilinear')


class MaskImageParams(ImageParams):

    def __init__(self, updateCallback: callable = None, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.updateCallback = updateCallback

        self.colormap_combo.setCurrentText('jet')
        self.interp_combo.setCurrentText('nearest')

        layout3 = qtw.QHBoxLayout()
        alpha_l = qtw.QLabel('alpha')
        self.alpha = MyQDoubleSpinBox()
        self.alpha.setRange(0, 1)
        self.alpha.setSingleStep(0.05)
        self.alpha.setValue(0.5)
        exclude_l = qtw.QLabel('except')
        self.exclude = EditableComboBox()
        exclude = ['None', 'min', 'max', 'blow(0)', 'above(1)']
        self.exclude.addItems(exclude)
        self.exclude.setCurrentText('None')
        self.addwidgets(
            layout3,
            [alpha_l, self.alpha, exclude_l, self.exclude],
        )

        self.ilayout.addLayout(layout3)

        self.vmin_input.editingFinished.connect(
            lambda: self.on_vmin_changed(self.vmin_input.text()))
        self.vmax_input.editingFinished.connect(
            lambda: self.on_vmax_changed(self.vmax_input.text()))
        self.colormap_combo.changed.connect(self.on_cmap_changed)
        self.interp_combo.currentTextChanged.connect(self.on_interp_changed)
        self.alpha.changed.connect(self.on_alpha_changed)
        self.exclude.changed.connect(self.on_except_changed)

    def set_callback(self, updateCallback: callable):
        self.updateCallback = updateCallback
        self.on_vmin_changed(self.vmin_input.text(), True)
        self.on_vmax_changed(self.vmax_input.text(), True)
        self.on_cmap_changed(self.colormap_combo.currentText(), True)
        self.on_interp_changed(self.interp_combo.currentText(), True)
        self.on_alpha_changed(self.alpha.value(), True)
        self.on_except_changed(self.exclude.currentText(), True)

    def on_vmin_changed(self, text, init=False):
        if self.updateCallback:
            idx = -1 if init else None
            self.updateCallback('vmin', float(text), idx)

    def on_vmax_changed(self, text, init=False):
        if self.updateCallback:
            idx = -1 if init else None
            self.updateCallback('vmax', float(text), idx)

    def on_cmap_changed(self, text, init=False):
        if self.updateCallback:
            idx = -1 if init else None
            self.updateCallback('cmap', text, idx)

    def on_interp_changed(self, text, init=False):
        if self.updateCallback:
            idx = -1 if init else None
            self.updateCallback('interp', text, idx)

    def on_alpha_changed(self, alpha, init=False):
        if self.updateCallback:
            idx = -1 if init else None
            self.updateCallback('alpha', float(alpha), idx)

    def on_except_changed(self, text, init=False):
        if self.updateCallback:
            idx = -1 if init else None
            self.updateCallback('except', text, idx)


class ItemsWidget(qtw.QWidget):
    # TODO: 改变顺序的时候发送信号
    params = QtCore.pyqtSignal(list)
    deleteIdx = QtCore.pyqtSignal(int)

    def __init__(self, visable_action=False, parent=None):
        super().__init__(parent)
        self.visable_action = visable_action
        self.ilayout = qtw.QVBoxLayout(self)

        self.listWidget = qtw.QListWidget()
        self.listWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.listWidget.customContextMenuRequested.connect(
            self.showContextMenu)
        self.listWidget.setDragDropMode(qtw.QListWidget.InternalMove)
        self.listWidget.itemClicked.connect(self.showDetails)
        self.ilayout.addWidget(self.listWidget, 1)
        self.ilayout.setContentsMargins(0, 0, 0, 0)
        self.ilayout.setSpacing(0)

        self.currentParamsWidget = None

    def addItem(self, item: qtw.QListWidgetItem):
        self.listWidget.addItem(item)
        item.paramsWidget.set_callback(self.updateCallback)

    def removeSelectedItem(self):
        item = self.listWidget.currentItem()
        idx = self.listWidget.currentRow()
        if item:
            paramsWidget = item.paramsWidget
            if paramsWidget == self.currentParamsWidget:
                self.ilayout.removeWidget(paramsWidget)
                paramsWidget.setParent(None)
                del paramsWidget
                self.currentParamsWidget = None
            self.listWidget.takeItem(self.listWidget.row(item))
            del item
            self.deleteIdx.emit(idx)

    def showDetails(self, item):
        if self.currentParamsWidget is not None:
            self.ilayout.removeWidget(self.currentParamsWidget)
            self.currentParamsWidget.hide()
        self.currentParamsWidget = item.paramsWidget
        self.ilayout.addWidget(self.currentParamsWidget, 2)  # 参数部分占用更多的空间
        self.currentParamsWidget.show()

    def showContextMenu(self, position):
        menu = qtw.QMenu()
        if self.visable_action:
            visbaleAction = menu.addAction("Visable")
        removeAction = menu.addAction("Delete")
        action = menu.exec_(self.listWidget.mapToGlobal(position))
        if action == removeAction:
            self.removeSelectedItem()
        if self.visable_action:
            if action == visbaleAction:
                pass

    def clear(self):
        # 清空所有项目和相关的参数界面
        while self.listWidget.count() > 0:
            item = self.listWidget.takeItem(0)
            if item.paramsWidget:
                item.paramsWidget.setParent(None)
                del item.paramsWidget
            del item  # 显式删除项目

    def updateCallback(self, mode: str, value, idx=None):
        # assert mode in ['vmin', 'vmax', 'cmap', 'interp', 'alpha', 'except']
        if idx is None:
            idx = self.listWidget.currentRow() # TODO: The idx is error when select the item
        self.params.emit([idx, mode, value])


class LoadFolder(qtw.QPushButton):
    fileList = QtCore.pyqtSignal(list)

    def __init__(self, parent=None):
        super(LoadFolder, self).__init__("Folder", parent)
        self.clicked.connect(self.loadFolder)
        self.loaded = False

    def loadFolder(self, filePath: str = None):
        if self.loaded:
            qtw.QMessageBox.critical(
                self, "Warn",
                "Folder already loaded. Please clear existing folder first.")
            return

        if not filePath:
            filePath = qtw.QFileDialog.getExistingDirectory(
                self,
                "Select Folder",
                "",
                # options=qtw.QFileDialog.DontUseNativeDialog
            )
        if not filePath:  # 用户取消选择
            return

        flist = list(Path(filePath).glob('*.dat'))
        flist += list(Path(filePath).glob('*.npy'))
        flist = [f for f in flist if not f.name.startswith('.')]
        flist = _sortfile(flist)
        n = len(flist)
        if n > 0:
            self.loaded = True
            self.fileList.emit(flist)
        else:
            qtw.QMessageBox.critical(self, "Warn", f"An empty folder.")
            return

    def clear(self):
        self.loaded = False


class LoadFolderWidget(qtw.QWidget):
    currentPath = QtCore.pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.flist = []

        hlayout = qtw.QHBoxLayout()
        self.loadFd = LoadFolder(self)
        self.fidx = MyQSpinBox()
        self.fname = qtw.QLineEdit("None")
        self.fname.setReadOnly(True)
        hlayout.addWidget(self.loadFd)
        hlayout.addWidget(self.fidx)
        hlayout.addWidget(self.fname)
        self.setLayout(hlayout)

        self.loadFd.fileList.connect(self.set_range)
        self.loadFd.fileList.connect(self.get_flist)
        self.fidx.changed.connect(self.set_fname)
        self.fidx.changed.connect(self.send_path)

    def set_range(self, flist):
        n = len(flist)
        self.fidx.setRange(0, n - 1)

    def get_flist(self, flist):
        self.flist = flist
        self.set_fname(self.fidx.value())
        self.send_path(self.fidx.value())

    def set_fname(self, idx):
        if len(self.flist) == 0:
            return

        idx = int(idx)
        self.fname.setText(Path(self.flist[idx]).name)

    def send_path(self, idx):
        if len(self.flist) == 0:
            return
        idx = int(idx)
        self.currentPath.emit(str(self.flist[idx]))

    def clear(self):
        self.loadFd.clear()
        self.fidx.setValue(0)
        self.fname.setText("None")
        self.flist = []
