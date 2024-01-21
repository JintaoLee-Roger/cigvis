# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Run: 
    python -c "import cigvis; cigvis.gui.gui3d()"

TODO:
    add other nodes (i.e., surface, well logs, ...)
"""

import re
import sys
import typing
from PyQt5.QtWidgets import QWidget
import numpy as np

from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore, QtGui

from vispy.app import use_app

import cigvis
from cigvis.vispynodes import VisCanvas

from .custom_widgets import *

CANVAS_SIZE = (800, 600)  # (width, height)

INT_validator = QtCore.QRegExp(r"^[1-9][0-9]*$")
FLOAT_validator = QtCore.QRegExp(r"[-+]?[0-9]*\.?[0-9]+")


class MyMainWindow(qtw.QMainWindow):

    def __init__(self, nx=None, ny=None, nz=None, clear_dim=True):
        super().__init__()

        self.clear_dim = clear_dim

        self.initUI(nx, ny, nz)

    def initUI(self, nx=None, ny=None, nz=None):
        central_widget = qtw.QWidget()
        self.main_layout = qtw.QHBoxLayout()

        self._controls = Controls(nx, ny, nz, self.clear_dim)
        self.main_layout.addWidget(self._controls)
        self._canvas_wrapper = CanvasWrapper()
        self.main_layout.addWidget(self._canvas_wrapper.canvas.native)

        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

        self._canvas_wrapper.canvas.native.setAcceptDrops(True)
        self._canvas_wrapper.canvas.native.dragEnterEvent = self.handleDragEnterEvent
        self._canvas_wrapper.canvas.native.dropEvent = self.handleDropEvent

        self._connect_controls()

    def handleDragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def handleDropEvent(self, event):
        if len(event.mimeData().urls()) > 1:
            qtw.QMessageBox.critical(self, "Error", "Only support 1 file")
            return
        fpath = event.mimeData().urls()[0].toLocalFile()
        self._controls.load_data(fpath)

    def _connect_controls(self):
        self._controls.data_loaded[list].connect(self._canvas_wrapper.set_data)
        self._controls.colormap_combo.changed.connect(self._set_cmap)
        self._controls.vmin_input.editingFinished.connect(
            lambda: self._canvas_wrapper.set_vmin(self._controls.vmin_input.
                                                  text()))
        self._controls.vmax_input.editingFinished.connect(
            lambda: self._canvas_wrapper.set_vmax(self._controls.vmax_input.
                                                  text()))

        self._controls.azimuth_input.changed.connect(
            self._canvas_wrapper.set_azimuth)
        self._controls.elevation_input.changed.connect(
            self._canvas_wrapper.set_elevation)
        self._controls.fov_input.changed.connect(self._canvas_wrapper.set_fov)

        self._controls.xpos.changed.connect(self._canvas_wrapper.set_xpos)
        self._controls.ypos.changed.connect(self._canvas_wrapper.set_ypos)
        self._controls.zpos.changed.connect(self._canvas_wrapper.set_zpos)

        self._controls.aspx.changed.connect(self._canvas_wrapper.set_aspectx)
        self._controls.aspy.changed.connect(self._canvas_wrapper.set_aspecty)
        self._controls.aspz.changed.connect(self._canvas_wrapper.set_aspectz)

        self._canvas_wrapper.camera_params[list].connect(
            self._controls.update_camera)
        self._controls.update_btn.clicked.connect(
            lambda: self._controls.update_camera(self._canvas_wrapper.
                                                 get_params()))
        self._controls.clear_btn.clicked.connect(self._canvas_wrapper.clear)
        self._controls.clear_btn.clicked.connect(self._controls.clear)

    def _set_cmap(self, cmap):
        try:
            self._canvas_wrapper.set_cmap(cmap)
        except Exception as e:
            qtw.QMessageBox.critical(self, "Error", f"Error colormap: {e}")


class Controls(qtw.QWidget):

    data_loaded = QtCore.pyqtSignal(list)

    def __init__(self, nx=None, ny=None, nz=None, clear_dim=True, parent=None):
        super().__init__(parent)

        self.clear_dim = clear_dim

        layout = qtw.QVBoxLayout()

        self.loaded = False

        # dimensions
        row1_layout = qtw.QHBoxLayout()
        nx_label = qtw.QLabel('nx:')
        self.nx_input = qtw.QLineEdit()
        self.nx_input.setValidator(QtGui.QRegExpValidator(INT_validator, self))
        if nx is not None:
            self.nx_input.setText(f'{nx}')
        ny_label = qtw.QLabel('ny:')
        self.ny_input = qtw.QLineEdit()
        self.ny_input.setValidator(QtGui.QRegExpValidator(INT_validator, self))
        if ny is not None:
            self.ny_input.setText(f'{ny}')
        nz_label = qtw.QLabel('nz:')
        self.nz_input = qtw.QLineEdit()
        self.nz_input.setValidator(QtGui.QRegExpValidator(INT_validator, self))
        if nz is not None:
            self.nz_input.setText(f'{nz}')
        self.load_btn = qtw.QPushButton('Load')
        self.addwidgets(row1_layout, [
            nx_label, self.nx_input, ny_label, self.ny_input, nz_label,
            self.nz_input, self.load_btn
        ])

        # clim
        row2_layout = qtw.QHBoxLayout()
        vmin_label = qtw.QLabel('vmin:')
        self.vmin_input = qtw.QLineEdit()
        self.vmin_input.setValidator(
            QtGui.QRegExpValidator(FLOAT_validator, self))
        vmax_label = qtw.QLabel('vmax:')
        self.vmax_input = qtw.QLineEdit()
        self.vmax_input.setValidator(
            QtGui.QRegExpValidator(FLOAT_validator, self))
        self.addwidgets(
            row2_layout,
            [vmin_label, self.vmin_input, vmax_label, self.vmax_input])

        # colormap
        row3_layout = qtw.QHBoxLayout()
        colormap_label = qtw.QLabel('Colormap:')
        self.colormap_combo = EditableComboBox()
        colormaps = [
            'gray', 'seismic', 'Petrel', 'od_seismic1', 'od_seismic2',
            'od_seismic3'
        ]
        self.colormap_combo.addItems(colormaps)
        self.colormap_combo.setCurrentText('gray')  # 默认值为'gray'
        self.addwidgets(row3_layout, [colormap_label, self.colormap_combo])

        # parameters of the camera
        row4_layout = qtw.QHBoxLayout()
        azimuth_label = qtw.QLabel('Azimuth:')
        self.azimuth_input = MyQDoubleSpinBox()
        self.azimuth_input.setRange(0, 360)
        self.azimuth_input.setValue(50)
        elevation_label = qtw.QLabel('Elevation:')
        self.elevation_input = MyQDoubleSpinBox()
        self.elevation_input.setRange(-90, 90)
        self.elevation_input.setValue(50)
        fov_label = qtw.QLabel('FOV:')
        self.fov_input = MyQDoubleSpinBox()
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
        ypos_label = qtw.QLabel('y:')
        self.ypos = MyQSpinBox()
        self.ypos.setMinimum(0)
        zpos_label = qtw.QLabel('z:')
        self.zpos = MyQSpinBox()
        self.zpos.setMinimum(0)

        self.addwidgets(row5_layout, [
            xpos_label, self.xpos, ypos_label, self.ypos, zpos_label, self.zpos
        ])

        # aspect ratio
        row_aspect = qtw.QHBoxLayout()
        asp1 = qtw.QLabel('Aspect Ratio  x ')
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

        self.addlayout(layout, [
            row1_layout, row2_layout, row3_layout, row4_layout, row5_layout,
            row_aspect, row6_layout
        ])

        layout.addStretch(1)
        self.setLayout(layout)
        self.setMaximumWidth(400)

        self.load_btn.clicked.connect(self.load_data)

    def addwidgets(self, layout, widgets):
        for widget in widgets:
            layout.addWidget(widget)

    def addlayout(self, layout, sublayouts):
        for sublayout in sublayouts:
            layout.addLayout(sublayout)

    def is_set_dim(self):
        if self.nx_input.text() and self.ny_input.text(
        ) and self.nz_input.text():
            return True
        else:
            return False

    def load_data(self, file_path=None):
        if self.loaded:
            qtw.QMessageBox.critical(self, "Warn",
                                     "Need to click clear to reset")
            return

        if not file_path:
            file_dialog = qtw.QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(self, 'Open Data File',
                                                       '', 'Binary Files (*)')

        if file_path and not self.is_set_dim():
            dim = get_dim_from_filename(file_path, False)
            if dim:
                self.nx_input.setText(dim[0])
                self.ny_input.setText(dim[1])
                self.nz_input.setText(dim[2])
            else:
                qtw.QMessageBox.critical(
                    self, "Error", "Please enter values for nx, ny, and nz.")
                return

        if file_path:
            try:
                nx = int(self.nx_input.text())
                ny = int(self.ny_input.text())
                nz = int(self.nz_input.text())

                if file_path.endswith('.vds'):
                    data = cigvis.io.VDSReader(file_path)
                else:
                    data = np.fromfile(file_path,
                                       dtype=np.float32).reshape(nx, ny, nz)

                if not self.vmin_input.text():
                    self.vmin_input.setText(f'{data.min():.2f}')
                if not self.vmax_input.text():
                    self.vmax_input.setText(f'{data.max():.2f}')

                self.xpos.setMaximum(nx - 1)
                self.ypos.setMaximum(ny - 1)
                self.zpos.setMaximum(nz - 1)

                if self.zpos.value() == 0:
                    self.zpos.setValue(nz - 1)

                self.data_loaded.emit([
                    data,
                    [self.xpos.value(),
                     self.ypos.value(),
                     self.zpos.value()],
                    [
                        float(self.vmin_input.text()),
                        float(self.vmax_input.text())
                    ],
                    self.colormap_combo.currentText(),
                    self.azimuth_input.value(),
                    self.elevation_input.value(),
                    self.fov_input.value(),
                    [self.aspx.value(),
                     self.aspy.value(),
                     self.aspz.value()]
                ])
                self.loaded = True
            except Exception as e:
                qtw.QMessageBox.critical(self, "Error",
                                         f"Error loading data: {e}")

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

    def clear(self):
        if self.clear_dim:
            self.nx_input.clear()
            self.ny_input.clear()
            self.nz_input.clear()
        self.vmin_input.clear()
        self.vmax_input.clear()
        self.colormap_combo.setCurrentText('gray')
        self.azimuth_input.setValue(50)
        self.elevation_input.setValue(50)
        self.fov_input.setValue(30)
        self.xpos.setValue(0)
        self.ypos.setValue(0)
        self.zpos.setValue(0)
        self.aspx.setValue(1)
        self.aspy.setValue(1)
        self.aspz.setValue(1)
        self.loaded = False


class CanvasWrapper(qtw.QWidget):

    camera_params = QtCore.pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.nodes = []
        self.canvas = VisCanvas(size=CANVAS_SIZE)
        self.canvas.events.key_press.connect(self.on_key_press)

    def set_data(self, data):
        self.data = data[0]
        self.canvas.update_camera(data[4], data[5], data[6])
        self.canvas.update_axis_scales(data[7])
        self.nodes = cigvis.create_slices(self.data,
                                          pos=data[1],
                                          clim=data[2],
                                          cmap=data[3])
        self.canvas.add_nodes(self.nodes)

    def set_cmap(self, cmap):
        cmap = cigvis.colormap.cmap_to_vispy(cmap)

        for node in self.nodes:
            node.cmap = cmap

    def set_vmin(self, vmin):
        vmin = float(vmin)
        for node in self.nodes:
            old_clim = node.clim
            node.clim = [vmin, old_clim[1]]

    def set_vmax(self, vmax):
        vmax = float(vmax)
        for node in self.nodes:
            old_clim = node.clim
            node.clim = [old_clim[0], vmax]

    def set_azimuth(self, azimuth):
        if not hasattr(self.canvas, 'view'):
            return
        for view in self.canvas.view:
            view.camera.azimuth = azimuth

    def set_elevation(self, elevation):
        if not hasattr(self.canvas, 'view'):
            return
        for view in self.canvas.view:
            view.camera.elevation = elevation

    def set_fov(self, fov):
        if not hasattr(self.canvas, 'view'):
            return
        for view in self.canvas.view:
            view.camera.fov = fov

    def set_xpos(self, xpos):
        if not xpos:
            return
        xpos = int(xpos)
        if len(self.nodes) > 0:
            for node in self.nodes:
                if node.axis == 'x':
                    node._update_location(xpos)

    def set_ypos(self, ypos):
        if not ypos:
            return
        ypos = int(ypos)
        if len(self.nodes) > 0:
            for node in self.nodes:
                if node.axis == 'y':
                    node._update_location(ypos)

    def set_zpos(self, zpos):
        if not zpos:
            return
        zpos = int(zpos)
        if len(self.nodes) > 0:
            for node in self.nodes:
                if node.axis == 'z':
                    node._update_location(zpos)

    def set_aspectx(self, aspx):
        r = cigvis.is_x_reversed()
        aspx *= (1 - 2 * r)
        if not hasattr(self.canvas, 'view'):
            return
        for view in self.canvas.view:
            axis_scales = view.camera._flip_factors
            axis_scales[0] = aspx
            view.camera._flip_factors = axis_scales
            view.camera._update_camera_pos()
            self.canvas.update()

    def set_aspecty(self, aspy):
        r = cigvis.is_y_reversed()
        aspy *= (1 - 2 * r)
        if not hasattr(self.canvas, 'view'):
            return
        for view in self.canvas.view:
            axis_scales = view.camera._flip_factors
            axis_scales[1] = aspy
            view.camera._flip_factors = axis_scales
            view.camera._update_camera_pos()
            self.canvas.update()

    def set_aspectz(self, aspz):
        r = cigvis.is_z_reversed()
        aspz *= (1 - 2 * r)
        if not hasattr(self.canvas, 'view'):
            return
        for view in self.canvas.view:
            axis_scales = view.camera._flip_factors
            axis_scales[2] = aspz
            view.camera._flip_factors = axis_scales
            view.camera._update_camera_pos()
            self.canvas.update()

    def get_params(self):
        out = None
        if hasattr(self.canvas, 'view'):
            camera = self.canvas.view[0].camera
            out = [camera.azimuth, camera.elevation, camera.fov]

        if len(self.nodes) > 0:
            pos = {'x': [], 'y': [], 'z': []}
            for node in self.nodes:
                axis = node.axis
                apos = node.pos
                pos[axis].append(apos)

            out.append(pos)

        return out

    def on_key_press(self, event):
        if event.key == 'u':
            out = self.get_params()
            if out is not None:
                self.camera_params.emit(out)

    def clear(self):
        for node in self.nodes:
            node.parent = None
            del node
        self.nodes = []
        del self.data
        self.data = None


class DimsWidget(qtw.QWidget):

    def __init__(self, nx=None, ny=None, nz=None, clear_dim=True, parent=None):
        super().__init__(parent)

        self.clear_dim = clear_dim

        # dimensions
        row1_layout = qtw.QHBoxLayout()
        nx_label = qtw.QLabel('nx:')
        self.nx_input = qtw.QLineEdit()
        self.nx_input.setValidator(QtGui.QRegExpValidator(INT_validator, self))
        if nx is not None:
            self.nx_input.setText(f'{nx}')
        ny_label = qtw.QLabel('ny:')
        self.ny_input = qtw.QLineEdit()
        self.ny_input.setValidator(QtGui.QRegExpValidator(INT_validator, self))
        if ny is not None:
            self.ny_input.setText(f'{ny}')
        nz_label = qtw.QLabel('nz:')
        self.nz_input = qtw.QLineEdit()
        self.nz_input.setValidator(QtGui.QRegExpValidator(INT_validator, self))
        if nz is not None:
            self.nz_input.setText(f'{nz}')
        self.load_btn = qtw.QPushButton('Load')
        self.addwidgets(row1_layout, [
            nx_label, self.nx_input, ny_label, self.ny_input, nz_label,
            self.nz_input, self.load_btn
        ])

    def addwidgets(self, layout, widgets):
        for widget in widgets:
            layout.addWidget(widget)

    def set_dims(self, nx, ny, nz):
        self.nx_input.setText(f'{nx}')
        self.ny_input.setText(f'{ny}')
        self.nz_input.setText(f'{nz}')

    def clear(self):
        if self.clear_dim:
            self.nx_input.clear()
            self.ny_input.clear()
            self.nz_input.clear()


def get_dim_from_filename(fname: str, return_int: bool = True):
    """
    obtain the dimension size from file path, 
    support template:
    - fname_h{z}x{y}x{x}.siff, i.e.,  fname_h128x500x200.dat
    """
    if fname.endswith(".vds"):
        vds = cigvis.io.VDSReader(fname)
        shape = vds.shape
        vds.close()
        return str(shape[0]), str(shape[1]), str(shape[2])
    
    f = fname.split('/')[-1]
    pattern = r'^[A-Za-z0-9_]+_h(\d+)x(\d+)x(\d+)+\.\w+$'
    m = re.match(pattern, f)
    if m:
        if return_int:
            return int(m.group(3)), int(m.group(2)), int(m.group(1))
        else:
            return m.group(3), m.group(2), m.group(1)
    else:
        return False


def gui3d(nx=None, ny=None, nz=None, clear_dim=True):
    app = use_app("pyqt5")
    app.create()
    win = MyMainWindow(nx, ny, nz, clear_dim)
    win.show()
    app.run()
