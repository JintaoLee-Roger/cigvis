# Copyright (c) 2024 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.

import platform
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore

from vispy.app import use_app

from .control_panel import ControlP, GlobalState
from .plot_canvas import PlotCanvas

from cigvis.gui.custom_widgets import *


class CentralController(QtCore.QObject):

    def __init__(self, controlPanel: ControlP, plotCanvas: PlotCanvas):
        self.controlP = controlPanel
        self.pCanvas = plotCanvas
        self.setupConnections()

    def setupConnections(self):
        self.controlP.loadBtn.data[np.ndarray].connect(self.pCanvas.set_data)
        self.cameraConnection()
        self.baseTabConnection()
        self.controlP.clear_btn.clicked.connect(self.pCanvas.clear)
        self.maskTabConnection()
        self.horzTabConnect()

    # fmt: off
    def cameraConnection(self):
        self.controlP.azimuth_input.changed.connect(self.pCanvas.set_azimuth)
        self.controlP.elevation_input.changed.connect(self.pCanvas.set_elevation)
        self.controlP.fov_input.changed.connect(self.pCanvas.set_fov)

        self.controlP.xpos.changed.connect(self.pCanvas.set_xpos)
        self.controlP.ypos.changed.connect(self.pCanvas.set_ypos)
        self.controlP.zpos.changed.connect(self.pCanvas.set_zpos)

        self.controlP.aspx.changed.connect(self.pCanvas.set_aspectx)
        self.controlP.aspy.changed.connect(self.pCanvas.set_aspecty)
        self.controlP.aspz.changed.connect(self.pCanvas.set_aspectz)

        # self.pCanvas.camera_params[list].connect(self.controlP.update_camera)
        self.controlP.update_btn.clicked.connect(lambda: self.controlP.update_camera(self.pCanvas.get_params()))

    def baseTabConnection(self):
        self.controlP.base_tab.colormap_combo.changed.connect(self.pCanvas.set_cmap)
        self.controlP.base_tab.interp_combo.currentTextChanged.connect(self.pCanvas.set_interp)
        self.controlP.base_tab.vmin_input.editingFinished.connect(lambda: self.pCanvas.set_vmin(self.controlP.base_tab.vmin_input.text()))
        self.controlP.base_tab.vmax_input.editingFinished.connect(lambda: self.pCanvas.set_vmax(self.controlP.base_tab.vmax_input.text()))

    def maskTabConnection(self):
        self.controlP.mask_tab.params[list].connect(self.pCanvas.set_mask_params)
        self.controlP.mask_tab.deleteIdx[int].connect(self.pCanvas.remove_mask)

    def horzTabConnect(self):
        self.controlP.horz_tab.params[list].connect(self.pCanvas.set_horz_params)
        self.controlP.horz_tab.deleteIdx[int].connect(self.pCanvas.remove_horz)
        self.controlP.horz_tab.light.selectionChanged[str].connect(self.pCanvas.canvas.update_light)

    # fmt: on


class MyMainWindow(qtw.QMainWindow):

    def __init__(self, nx=None, ny=None, nz=None, clear_dim=True):
        super().__init__()

        self.clear_dim = clear_dim
        self.gstates = GlobalState(nx, ny, nz)

        self.initUI(nx, ny, nz)

    def initUI(self, nx=None, ny=None, nz=None):
        central_widget = qtw.QWidget()
        self.main_layout = qtw.QHBoxLayout()

        self.controlP = ControlP(self.gstates, nx, ny, nz, self.clear_dim)
        self.main_layout.addWidget(self.controlP)

        self.plotCanvas = PlotCanvas(parent=self)
        self.main_layout.addWidget(self.plotCanvas)
        self.centerControl = CentralController(self.controlP, self.plotCanvas)

        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()


def gui3d(nx=None, ny=None, nz=None, clear_dim=True, data=None):
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    system = platform.system()

    app = use_app("pyqt5")
    app.create()
    win = MyMainWindow(nx, ny, nz, clear_dim)
    if data is not None:
        win.controlP.loadBtn.sent_data(data)

    if system == 'Linux':
        font = QtGui.QFont('Ubuntu')
        win.setFont(font)
    elif system == 'Windows':
        font = QtGui.QFont('Segoe UI')
        win.setFont(font)

    win.show()
    app.run()