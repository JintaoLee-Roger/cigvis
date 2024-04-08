# Copyright (c) 2024 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.

import sys
import re
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore, QtGui
from pathlib import Path
import cigvis

from cigvis.gui.custom_widgets import *

INTERPS = [
    'nearest', 'linear', 'bicubic', 'bilinear', 'cubic', 'sinc', 'blackman',
    'catrom', 'bessel', 'gaussian', 'hamming', 'hanning', 'hermite', 'kaiser',
    'lanczos', 'mitchell', 'quadric', 'spline16', 'spline36'
]


class GlobalState(QtCore.QObject):

    def __init__(self, nx=None, ny=None, nz=None) -> None:
        self.dataLoaded = False
        self.loadType = 'base'
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.transpose = False

    def get_shape(self):
        if (self.nx is None) or (self.ny is None) or (self.nz is None):
            return False
        return self.nx, self.ny, self.nz


class LoadBtn(qtw.QPushButton):
    data = QtCore.pyqtSignal(object)  # 用于通知数据加载完成的信号
    nx = QtCore.pyqtSignal(str)
    ny = QtCore.pyqtSignal(str)
    nz = QtCore.pyqtSignal(str)
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

        if filePath and not self.gstates.get_shape():
            dim = _get_dim_from_filename(filePath)
            if dim:
                nx, ny, nz = dim
                self.gstates.nx = nx
                self.gstates.ny = ny
                self.gstates.nz = nz
                self.nx.emit(f'{nx}')
                self.ny.emit(f'{ny}')
                self.nz.emit(f'{nz}')
            else:
                qtw.QMessageBox.critical(
                    self, "Error", "Please enter values for nx, ny, and nz.")
                return

        if filePath:
            nx, ny, nz = self.gstates.get_shape()
            try:
                if filePath.endswith('.vds'):
                    data = cigvis.io.VDSReader(filePath)
                elif filePath.endswith('.npy'):
                    data = np.load(filePath)
                else:
                    # data = np.memmap(filePath,
                    #                  np.float32,
                    #                  'c',
                    #                  shape=(nx, ny, nz))
                    data = np.fromfile(filePath,
                                       np.float32).reshape(nx, ny, nz)
                if not self._is_base():
                    nxc, nyc, nzc = data.shape
                    if (nxc != nx) or (nyc != ny) or (nzc != nz):
                        qtw.QMessageBox.critical(
                            self, "Warn",
                            f"Mask image's shape must be same as base image, but base is ({nx}, {ny}, {nz}), mask is ({nxc}, {nyc}, {nzc})") # yapf: disable
                        return
            except Exception as e:
                qtw.QMessageBox.critical(self, "Error",
                                         f"Error loading data: {e}")

        if self.gstates.transpose:
            data = data.T
        self.gstates.dataLoaded = True  # 标记数据已加载
        if self._is_base():
            self.vmin.emit(f'{data.min():.2f}')
            self.vmax.emit(f'{data.max():.2f}')
        else:
            item = qtw.QListWidgetItem(Path(filePath).name)
            paramsWidget = MaskImageParams(interps=INTERPS)
            paramsWidget.vmin_input.setTextAndEmit(f'{data.min():.2f}')
            paramsWidget.vmax_input.setTextAndEmit(f'{data.max():.2f}')
            item.paramsWidget = paramsWidget
            item.visible = True  # TODO:
            self.maskItem.emit(item)

        self.data.emit(data)  # 发送数据加载完成的信号


class HorizonWidget(qtw.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)


class ControlP(qtw.QWidget):

    def __init__(self,
                 gstates: GlobalState,
                 nx: int = None,
                 ny: int = None,
                 nz: int = None,
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
        nz_label = qtw.QLabel('nz:')
        self.nz_input = MyQLineEdit()
        self.nz_input.setValidator(QtGui.QRegExpValidator(INT_validator, self))
        if nz is not None:
            self.nz_input.setTextAndEmit(f'{nz}')
        self.addwidgets(row1_layout, [
            nx_label, self.nx_input, ny_label, self.ny_input, nz_label,
            self.nz_input
        ])

        row_folder = qtw.QHBoxLayout()
        self.loadfolder = LoadFolderWidget()
        self.addwidgets(row_folder, [self.loadfolder])

        row2_layout = qtw.QHBoxLayout()
        self.loadBtn = LoadBtn(gstates, self)
        trans_label = qtw.QLabel("Transpose")
        self.loadRad = RadioButtonPanel(['on', 'off'])
        self.loadRad.radioButtons[1].setChecked(True)
        self.addwidgets(row2_layout, [self.loadBtn, trans_label, self.loadRad])

        # parameters of the camera
        row4_layout = qtw.QHBoxLayout()
        azimuth_label = qtw.QLabel('Azimuth:')
        self.azimuth_input = MyQSpinBox()
        self.azimuth_input.setRange(0, 360)
        self.azimuth_input.setValue(50)
        elevation_label = qtw.QLabel('Elevation:')
        self.elevation_input = MyQSpinBox()
        self.elevation_input.setRange(-90, 90)
        self.elevation_input.setValue(50)
        fov_label = qtw.QLabel('FOV:')
        self.fov_input = MyQSpinBox()
        self.fov_input.setRange(1, 179)
        self.fov_input.setValue(30)
        self.addwidgets(row4_layout, [
            azimuth_label, self.azimuth_input, elevation_label,
            self.elevation_input, fov_label, self.fov_input
        ])

        # pos
        row5_layout = qtw.QHBoxLayout()
        xpos_label = qtw.QLabel('x:')
        self.xpos = MyQSpinBox()
        self.xpos.setMinimum(0)
        if nx:
            self.xpos.setMaximum(nx - 1)
        ypos_label = qtw.QLabel('y:')
        self.ypos = MyQSpinBox()
        self.ypos.setMinimum(0)
        if ny:
            self.ypos.setMaximum(ny - 1)
        zpos_label = qtw.QLabel('z:')
        self.zpos = MyQSpinBox()
        self.zpos.setMinimum(0)
        if nz:
            self.zpos.setMaximum(nz - 1)

        self.addwidgets(row5_layout, [
            xpos_label, self.xpos, ypos_label, self.ypos, zpos_label, self.zpos
        ])

        # aspect ratio
        row_aspect = qtw.QHBoxLayout()
        asp1 = qtw.QLabel('Aspect x ')
        self.aspx = MyQDoubleSpinBox()
        self.aspx.setMinimum(0.1)
        self.aspx.setValue(1)
        self.aspx.setSingleStep(0.1)
        asp2 = qtw.QLabel('  :  y ')
        self.aspy = MyQDoubleSpinBox()
        self.aspy.setMinimum(0.1)
        self.aspy.setSingleStep(0.1)
        self.aspy.setValue(1)
        asp3 = qtw.QLabel('  :  z ')
        self.aspz = MyQDoubleSpinBox()
        self.aspz.setMinimum(0.1)
        self.aspz.setSingleStep(0.1)
        self.aspz.setValue(1)

        self.addwidgets(row_aspect,
                        [asp1, self.aspx, asp2, self.aspy, asp3, self.aspz])

        # update params
        row6_layout = qtw.QHBoxLayout()
        self.update_btn = qtw.QPushButton('update params')
        self.clear_btn = qtw.QPushButton('clear')

        self.addwidgets(row6_layout, [self.update_btn, self.clear_btn])

        row_tab = qtw.QHBoxLayout()
        self.tab_widget = qtw.QTabWidget()
        self.base_tab = ImageParams(interps=INTERPS)
        self.tab_widget.addTab(self.base_tab, "Base")
        self.mask_tab = MaskWidget()
        self.tab_widget.addTab(self.mask_tab, "Masks")
        self.horz_tab = HorizonWidget()
        self.tab_widget.addTab(self.horz_tab, "Horiz")
        self.other_tab = HorizonWidget()
        self.tab_widget.addTab(self.other_tab, "Other")
        row_tab.addWidget(self.tab_widget)
        row_tab.setStretch(0, 1)

        self.addlayout(layout, [
            row1_layout,
            row_folder,
            row2_layout,
            row4_layout,
            row5_layout,
            row_aspect,
            row6_layout,
            row_tab,
        ])

        layout.addStretch(1)
        self.setLayout(layout)
        self.setMaximumWidth(350)
        self.innerConnection()

    def innerConnection(self):
        self.nx_input.editingFinished.connect(
            lambda: self.update_xpos_limit(self.nx_input.text()))
        self.ny_input.editingFinished.connect(
            lambda: self.update_ypos_limit(self.ny_input.text()))
        self.nz_input.editingFinished.connect(
            lambda: self.update_zpos_limit(self.nz_input.text()))

        self.nx_input.editingFinished.connect(
            lambda: self.update_nx(self.nx_input.text()))
        self.ny_input.editingFinished.connect(
            lambda: self.update_ny(self.ny_input.text()))
        self.nz_input.editingFinished.connect(
            lambda: self.update_nz(self.nz_input.text()))

        self.clear_btn.clicked.connect(self.clear)
        self.loadBtn.nx[str].connect(self.nx_input.setTextAndEmit)
        self.loadBtn.ny[str].connect(self.ny_input.setTextAndEmit)
        self.loadBtn.vmin[str].connect(self.base_tab.vmin_input.setTextAndEmit)
        self.loadBtn.vmax[str].connect(self.base_tab.vmax_input.setTextAndEmit)
        self.loadfolder.currentPath[str].connect(
            lambda fpath: self.loadBtn.loadData(fpath, check=False))
        self.loadBtn.maskItem[qtw.QListWidgetItem].connect(
            self.mask_tab.addItem)

        self.tab_widget.currentChanged.connect(self.tabSelected)
        self.loadRad.selectionChanged[str].connect(self.transpose)

    def tabSelected(self, idx):
        if idx == 1:
            # self.loadRad.radioButtons[1].setChecked(True)
            self.gstates.loadType = 'mask'
        else:
            # self.loadRad.radioButtons[0].setChecked(True)
            self.gstates.loadType = 'base'

    def addwidgets(self, layout, widgets):
        for widget in widgets:
            layout.addWidget(widget)

    def addlayout(self, layout, sublayouts):
        for sublayout in sublayouts:
            layout.addLayout(sublayout)

    def update_camera(self, params):
        if params is None:
            return
        self.azimuth_input.setValue(params[0])
        self.elevation_input.setValue(params[1])
        self.fov_input.setValue(params[2])

        if len(params) == 4:
            pos = params[3]
            self.xpos.setValue(pos['x'][0])
            self.ypos.setValue(pos['y'][0])
            self.zpos.setValue(pos['z'][0])

    def update_xpos_limit(self, nx):
        if nx:
            self.xpos.setMaximum(int(nx) - 1)

    def update_ypos_limit(self, ny):
        if ny:
            self.ypos.setMaximum(int(ny) - 1)

    def update_zpos_limit(self, nz):
        if nz:
            self.zpos.setMaximum(int(nz) - 1)

    def update_nx(self, nx):
        if nx:
            self.gstates.nx = int(nx)

    def update_ny(self, ny):
        if ny:
            self.gstates.ny = int(ny)

    def update_nz(self, nz):
        if nz:
            self.gstates.nz = int(nz)

    def transpose(self, text):
        if text == 'on':
            self.gstates.transpose = True
        else:
            self.gstates.transpose = False

    def clear(self):
        if self.clear_dim:
            self.nx_input.clear()
            self.ny_input.clear()
            self.nz_input.clear()
            self.gstates.nx = None
            self.gstates.ny = None
            self.gstates.nz = None

        self.gstates.dataLoaded = False
        self.gstates.loadType = 'base'

        self.loadfolder.clear()
        self.base_tab.clear()
        self.mask_tab.clear()

        self.azimuth_input.setValue(50)
        self.elevation_input.setValue(50)
        self.fov_input.setValue(30)
        self.xpos.setValue(0)
        self.ypos.setValue(0)
        self.zpos.setValue(0)
        self.aspx.setValue(1)
        self.aspy.setValue(1)
        self.aspz.setValue(1)


def _get_dim_from_filename(fname: str, return_int: bool = True):
    """
    obtain the dimension size from file path, 
    support template:
    - fname_h{z}x{y}x{x}.siff, e.g.,  fname_h128x500x200.dat
    - xxx.vds, i.e., VDS file
    - fname_{x}_{y}_{z}.sufix, e.g., fname_200_500_128.dat
    - xxx.npy, i.e., numpy file
    """
    if fname.endswith(".vds"):
        vds = cigvis.io.VDSReader(fname)
        shape = vds.shape
        vds.close()
        if return_int:
            return shape
        else:
            return str(shape[0]), str(shape[1]), str(shape[2])
    elif fname.endswith(".npy"):
        with open(fname, 'rb') as f:
            version = np.lib.format.read_magic(f)
            shape, _, _ = np.lib.format._read_array_header(f, version)

        if len(shape) != 3:
            return False

        if return_int:
            return shape
        else:
            return str(shape[0]), str(shape[1]), str(shape[2])

    pattern1 = r'\w+\_h(\d+)x(\d+)x(\d+)'
    pattern2 = r'\w+\_(\d+)\_(\d+)\_(\d+)'
    fname = Path(fname).stem
    match1 = re.search(pattern1, fname)
    if match1:
        z, y, x = match1.groups()
        if return_int:
            return int(x), int(y), int(z)
        else:
            return x, y, z

    match2 = re.search(pattern2, fname)
    if match2:
        x, y, z = match2.groups()
        if return_int:
            return int(x), int(y), int(z)
        else:
            return x, y, z

    return False


if __name__ == '__main__':
    fname = '/dsf/dsds/dsc_ds_123_412_512.dat'
    print(_get_dim_from_filename(fname))
