# Copyright (c) 2024 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.

import sys
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore, QtGui
from pathlib import Path

from cigvis.gui.custom_widgets import *


def _sortfile(flist: List[Path]):
    try:
        flist = sorted(flist, key=lambda x: int(x.stem))
    except:
        flist = sorted(flist)

    return flist


class GlobalState(QtCore.QObject):

    def __init__(self, nx=None, ny=None) -> None:
        self.dataLoaded = False
        self.loadType = 'base'
        self.nx = nx
        self.ny = ny

    def getNXNY(self):
        return self.nx, self.ny


class LoadBtn(qtw.QPushButton):
    data = QtCore.pyqtSignal(np.ndarray)  # 用于通知数据加载完成的信号
    nx = QtCore.pyqtSignal(str)
    ny = QtCore.pyqtSignal(str)
    vmin = QtCore.pyqtSignal(str)
    vmax = QtCore.pyqtSignal(str)
    maskItem = QtCore.pyqtSignal(qtw.QListWidgetItem)

    def __init__(self, gstates: GlobalState, parent=None):
        super(LoadBtn, self).__init__("Load File", parent)
        self.gstates = gstates
        self.clicked.connect(self.loadData)

    def _is_base(self):
        if self.gstates.loadType == 'base':
            return True
        else:
            return False

    def _check_load_valid(self):
        flag = True
        if self.gstates.dataLoaded and self._is_base():
            qtw.QMessageBox.critical(
                self, "Warn",
                "Data already loaded. Please clear existing data first.")
            flag = False

        if not self.gstates.dataLoaded and not self._is_base():
            qtw.QMessageBox.critical(
                self, "Warn",
                "Base Image is empty. Please load base image first.")
            flag = False
        return flag

    def loadData(self, filePath: str = None, check=True):
        if check and not self._check_load_valid():
            return

        if not filePath:
            filePath, _ = qtw.QFileDialog.getOpenFileName(
                self,
                "Open File",
                "",
                "All Files (*)",
                # options=qtw.QFileDialog.DontUseNativeDialog
            )
        if not filePath:  # 用户取消选择
            return

        # 检查文件格式并加载数据
        if filePath.endswith('.npy'):
            data = np.load(filePath)
            nx, ny = data.shape
            if self._is_base():
                self.gstates.nx = nx
                self.gstates.ny = ny
                self.nx.emit(f'{nx}')
                self.ny.emit(f'{ny}')
            else:
                nxc, nyc = self.gstates.getNXNY()
                if nxc != nx or nyc != ny:
                    qtw.QMessageBox.critical(
                        self, "Warn",
                        f"Mask image's shape must be same as base image, but base is ({nxc}, {nyc}), mask is ({nx}, {ny})") # yapf: disable
                    return
        else:
            nx, ny = self.gstates.getNXNY()
            if nx is None or ny is None:
                qtw.QMessageBox.critical(self, "Error",
                                         "Please enter values for nx, ny")
                return
            data = np.fromfile(filePath, np.float32).reshape(nx, ny)

        self.gstates.dataLoaded = True  # 标记数据已加载
        if self._is_base():
            self.vmin.emit(f'{data.min():.2f}')
            self.vmax.emit(f'{data.max():.2f}')
        else:
            item = qtw.QListWidgetItem(Path(filePath).name)
            paramsWidget = MaskImageParams()
            paramsWidget.vmin_input.setTextAndEmit(f'{data.min():.2f}')
            paramsWidget.vmax_input.setTextAndEmit(f'{data.max():.2f}')
            item.paramsWidget = paramsWidget
            self.maskItem.emit(item)

        self.data.emit(data)  # 发送数据加载完成的信号


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


class AnnotationWidget(qtw.QWidget):
    pospointS = QtCore.pyqtSignal(int)
    boxS = QtCore.pyqtSignal(int)
    brushS = QtCore.pyqtSignal(int)

    def __init__(self, gstates: GlobalState, parent=None):
        super(AnnotationWidget, self).__init__(parent)
        self.gstates = gstates

        grid_layout = qtw.QGridLayout()
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(0)
        self.hover_pos = ToggleButton("+")
        self.hover_neg = ToggleButton("-")
        self.hover_reset = qtw.QPushButton("reset")
        self.hover_undo = qtw.QPushButton('undo')

        self.box_add = ToggleButton("+")
        self.box_reset = qtw.QPushButton("reset")
        self.box_undo = qtw.QPushButton('undo')

        self.hover_btn = ToggleButton("Hover")
        self.box_btn = ToggleButton("Box")
        self.brush_btn = ToggleButton("Brush")
        self.brush_reset = qtw.QPushButton("reset")
        self.brush_undo = qtw.QPushButton("undo")
        size_label = qtw.QLabel('Brush Size')
        self.brush_size = qtw.QSlider(QtCore.Qt.Horizontal)
        self.brush_size.setMinimum(1)
        self.brush_size.setMaximum(100)
        self.brush_size.setValue(10)

        grid_layout.addWidget(self.hover_btn, 0, 0, 1, 2)
        grid_layout.addWidget(self.hover_pos, 1, 0)
        grid_layout.addWidget(self.hover_neg, 1, 1)
        grid_layout.addWidget(self.hover_reset, 2, 0)
        grid_layout.addWidget(self.hover_undo, 2, 1)

        grid_layout.addWidget(self.box_btn, 0, 2, 1, 2)
        grid_layout.addWidget(self.box_add, 1, 2, 1, 2)
        grid_layout.addWidget(self.box_reset, 2, 2)
        grid_layout.addWidget(self.box_undo, 2, 3)
        grid_layout.addWidget(self.brush_btn, 3, 0, 1, 2)
        grid_layout.addWidget(self.brush_reset, 3, 2)
        grid_layout.addWidget(self.brush_undo, 3, 3)
        grid_layout.addWidget(size_label, 4, 0, 1, 2)
        grid_layout.addWidget(self.brush_size, 4, 2, 1, 2)
        self.setLayout(grid_layout)
        self.connect_btn()

    def connect_btn(self):
        self.hover_btn.clicked.connect(self.active_marker)
        self.hover_pos.clicked.connect(self.active_marker)
        self.hover_neg.clicked.connect(self.active_marker)
        self.box_btn.clicked.connect(self.active_box)
        self.box_add.clicked.connect(self.active_box)
        self.brush_btn.clicked.connect(self.active_brush)

    def updateToggleState(self, toggledButton, checked):
        # 当一个按钮被切换到 ON，确保另一个按钮是 OFF
        if checked:
            if toggledButton == self.hover_btn:
                self.box_btn.setChecked(False)
                self.box_add.setChecked(False)
                self.brush_btn.setChecked(False)
            elif toggledButton == self.box_btn:
                self.hover_btn.setChecked(False)
                self.hover_pos.setChecked(False)
                self.hover_neg.setChecked(False)
                self.brush_btn.setChecked(False)
            elif toggledButton == self.brush_btn:
                self.box_btn.setChecked(False)
                self.box_add.setChecked(False)
                self.hover_btn.setChecked(False)
                self.hover_pos.setChecked(False)
                self.hover_neg.setChecked(False)
            elif toggledButton == self.hover_pos:
                self.hover_neg.setChecked(False)
                self.brush_btn.setChecked(False)
                self.hover_btn.setChecked(True)
            elif toggledButton == self.hover_neg:
                self.hover_pos.setChecked(False)
                self.brush_btn.setChecked(False)
                self.hover_btn.setChecked(True)
            elif toggledButton == self.box_add:
                self.box_btn.setChecked(True)
        else:
            if toggledButton == self.hover_btn:
                self.hover_pos.setChecked(False)
                self.hover_neg.setChecked(False)
            elif toggledButton == self.box_btn:
                self.box_add.setChecked(False)
            elif toggledButton == self.brush_btn:
                self.brush_btn.setChecked(False)

    def active_marker(self):
        if not self.gstates.dataLoaded:
            self.pospointS.emit(-1)
            return

        if not self.hover_btn.isChecked():
            self.pospointS.emit(-1)
            return
        else:
            self.boxS.emit(-1)
            self.brushS.emit(-1)

        if self.hover_pos.isChecked():
            self.pospointS.emit(1)
        elif self.hover_neg.isChecked():
            self.pospointS.emit(0)

    def active_box(self):
        if not self.gstates.dataLoaded:
            self.boxS.emit(-1)
            return
        if not self.box_btn.isChecked():
            self.boxS.emit(-1)
            return
        else:
            self.pospointS.emit(-1)
            self.brushS.emit(-1)

        if self.box_add.isChecked():
            self.boxS.emit(1)

    def active_brush(self):
        if not self.gstates.dataLoaded:
            self.brushS.emit(-1)
            return

        if not self.brush_btn.isChecked():
            self.brushS.emit(-1)
            return
        else:
            self.pospointS.emit(-1)
            self.boxS.emit(-1)
            self.brushS.emit(1)

    def clear(self):
        self.box_btn.setChecked(False)
        self.hover_btn.setChecked(False)
        self.brush_btn.setChecked(False)
        self.brush_size.setValue(10)


class ImageParams(BaseWidget):

    def __init__(self, parent=None) -> None:
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
            'gray', 'seismic', 'Petrel', 'stratum', 'jet', 'od_seismic1',
            'od_seismic2', 'od_seismic3'
        ]
        self.colormap_combo.addItems(colormaps)
        self.colormap_combo.setCurrentText('gray')  # 默认值为'gray'

        # interpolation
        interp_label = qtw.QLabel('Interp:')
        self.interp_combo = qtw.QComboBox()
        interps = [
            'none', 'nearest', 'bilinear', 'bicubic', 'quadric', 'sinc',
            'blackman', 'antialiased', 'spline36', 'mitchell', 'hamming',
            'catrom', 'gaussian', 'hanning', 'lanczos', 'bessel', 'spline16',
            'kaiser', 'hermite'
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

    def __init__(self, updateCallback: callable = None, parent=None):
        super().__init__(parent)
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
        self.on_vmin_changed(self.vmin_input.text())
        self.on_vmax_changed(self.vmax_input.text())
        self.on_cmap_changed(self.colormap_combo.currentText())
        self.on_interp_changed(self.interp_combo.currentText())
        self.on_alpha_changed(self.alpha.value())
        self.on_except_changed(self.exclude.currentText())

    def on_vmin_changed(self, text):
        if self.updateCallback:
            self.updateCallback('vmin', float(text))

    def on_vmax_changed(self, text):
        if self.updateCallback:
            self.updateCallback('vmax', float(text))

    def on_cmap_changed(self, text):
        if self.updateCallback:
            self.updateCallback('cmap', text)

    def on_interp_changed(self, text):
        if self.updateCallback:
            self.updateCallback('interp', text)

    def on_alpha_changed(self, alpha):
        if self.updateCallback:
            self.updateCallback('alpha', float(alpha))

    def on_except_changed(self, text):
        if self.updateCallback:
            self.updateCallback('except', text)


class MaskWidget(qtw.QWidget):
    params = QtCore.pyqtSignal(list)
    deleteIdx = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
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
        removeAction = menu.addAction("Delete")
        action = menu.exec_(self.listWidget.mapToGlobal(position))
        if action == removeAction:
            self.removeSelectedItem()

    def clear(self):
        # 清空所有项目和相关的参数界面
        while self.listWidget.count() > 0:
            item = self.listWidget.takeItem(0)
            if item.paramsWidget:
                item.paramsWidget.setParent(None)
                del item.paramsWidget
            del item  # 显式删除项目

    def updateCallback(self, mode: str, value):
        assert mode in ['vmin', 'vmax', 'cmap', 'interp', 'alpha', 'except']
        idx = self.listWidget.currentRow()
        self.params.emit([idx, mode, value])


class ControlP(qtw.QWidget):

    def __init__(self,
                 gstates: GlobalState,
                 nx: int = None,
                 ny: int = None,
                 clear_dim: bool = True,
                 parent=None):
        super().__init__(parent)
        assert isinstance(gstates, GlobalState)

        self.gstates = gstates
        self.clear_dim = clear_dim

        layout = qtw.QVBoxLayout()

        # dimensions
        row1_layout = qtw.QHBoxLayout()
        nx_label = qtw.QLabel('nx:')
        self.nx_input = MyQLineEdit()
        self.nx_input.setValidator(QtGui.QRegExpValidator(INT_validator, self))
        if nx is not None:
            self.nx_input.setTextAndEmit(f'{nx}')
        ny_label = qtw.QLabel('ny:')
        self.ny_input = MyQLineEdit()
        self.ny_input.setValidator(QtGui.QRegExpValidator(INT_validator, self))
        if ny is not None:
            self.ny_input.setTextAndEmit(f'{ny}')
        self.addwidgets(
            row1_layout,
            [nx_label, self.nx_input, ny_label, self.ny_input],
        )

        row_folder = qtw.QHBoxLayout()
        self.loadfolder = LoadFolderWidget()
        self.addwidgets(row_folder, [self.loadfolder])

        row2_layout = qtw.QHBoxLayout()
        self.loadBtn = LoadBtn(gstates, self)
        self.loadRad = RadioButtonPanel(['base', 'mask'])
        self.addwidgets(row2_layout, [self.loadBtn, self.loadRad])

        # clear and save
        rowl_layout = qtw.QHBoxLayout()
        self.save_btn = qtw.QPushButton('save')
        self.clear_btn = qtw.QPushButton('clear')
        self.addwidgets(rowl_layout, [self.save_btn, self.clear_btn])

        row_tab = qtw.QHBoxLayout()
        self.tab_widget = qtw.QTabWidget()
        self.base_tab = ImageParams()
        self.tab_widget.addTab(self.base_tab, "Base")
        self.anno_tab = AnnotationWidget(gstates)
        self.tab_widget.addTab(self.anno_tab, "Annotation")
        self.mask_tab = MaskWidget()
        self.tab_widget.addTab(self.mask_tab, "Masks")
        row_tab.addWidget(self.tab_widget)
        row_tab.setStretch(0, 1)

        self.addlayout(layout, [
            row1_layout,
            row_folder,
            row2_layout,
            rowl_layout,
            row_tab,
        ])

        layout.addStretch(1)
        self.setLayout(layout)
        self.setMaximumWidth(300)
        self.inner_connection()

    def inner_connection(self):
        self.clear_btn.clicked.connect(self.clear)
        self.loadBtn.nx[str].connect(self.nx_input.setTextAndEmit)
        self.loadBtn.ny[str].connect(self.ny_input.setTextAndEmit)
        self.loadBtn.vmin[str].connect(self.set_vmin)
        self.loadBtn.vmax[str].connect(self.set_vmax)
        self.loadfolder.currentPath[str].connect(
            lambda fpath: self.loadBtn.loadData(fpath, check=False))
        self.loadBtn.maskItem[qtw.QListWidgetItem].connect(
            self.mask_tab.addItem)

        self.tab_widget.currentChanged.connect(self.tabSelected)

    def addwidgets(self, layout, widgets):
        for widget in widgets:
            layout.addWidget(widget)

    def addlayout(self, layout, sublayouts):
        for sublayout in sublayouts:
            layout.addLayout(sublayout)

    def set_vmin(self, vmin):
        if self.gstates.loadType == 'base':
            self.base_tab.vmin_input.setTextAndEmit(vmin)
        else:
            pass

    def set_vmax(self, vmax):
        if self.gstates.loadType == 'base':
            self.base_tab.vmax_input.setTextAndEmit(vmax)
        else:
            pass

    def tabSelected(self, idx):
        if idx == 2:
            self.loadRad.radioButtons[1].setChecked(True)
            self.gstates.loadType = 'mask'
        else:
            self.loadRad.radioButtons[0].setChecked(True)
            self.gstates.loadType = 'base'

    def clear(self):
        if self.clear_dim:
            self.nx_input.clear()
            self.ny_input.clear()
            self.gstates.nx = None
            self.gstates.ny = None

        self.gstates.dataLoaded = False
        self.gstates.loadType = 'base'
        self.loadfolder.clear()
        self.base_tab.clear()
        self.anno_tab.clear()
        self.mask_tab.clear()
