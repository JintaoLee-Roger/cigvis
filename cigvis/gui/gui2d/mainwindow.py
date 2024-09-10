import sys
import platform
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore
from PyQt5.QtGui import QFont

import numpy as np

from .control_panel import ControlP, GlobalState
from .plot_canvas import PlotCanvas


class CentralController(QtCore.QObject):

    def __init__(self, controlPanel: ControlP, plotCanvas: PlotCanvas):
        self.controlP = controlPanel
        self.pCanvas = plotCanvas
        self.setupConnections()

    def setupConnections(self):
        self.loadDataConnection()
        self.baseTabConnection()
        self.annotateTabConnection()
        self.maskTabConnection()
        self.controlP.clear_btn.clicked.connect(self.pCanvas.clear)
        self.controlP.save_btn.clicked.connect(self.pCanvas.save_fig)

    def baseTabConnection(self):
        self.controlP.base_tab.colormap_combo.changed.connect(
            self.pCanvas.set_cmap)
        self.controlP.base_tab.interp_combo.currentTextChanged.connect(
            self.pCanvas.set_interp)
        self.controlP.base_tab.vmin_input.editingFinished.connect(
            lambda: self.pCanvas.set_vmin(self.controlP.base_tab.vmin_input.
                                          text()))
        self.controlP.base_tab.vmax_input.editingFinished.connect(
            lambda: self.pCanvas.set_vmax(self.controlP.base_tab.vmax_input.
                                          text()))

    def annotateTabConnection(self):
        # annotation
        self.controlP.anno_tab.pospointS[int].connect(
            self.pCanvas.set_marker_mode)
        self.controlP.anno_tab.boxS[int].connect(self.pCanvas.set_box_mode)
        self.controlP.anno_tab.brushS[int].connect(self.pCanvas.set_brush_mode)
        self.controlP.anno_tab.hover_reset.clicked.connect(
            self.pCanvas.marker_reset)
        self.controlP.anno_tab.hover_undo.clicked.connect(
            self.pCanvas.marker_undo)
        self.controlP.anno_tab.box_reset.clicked.connect(
            self.pCanvas.box_reset)
        self.controlP.anno_tab.box_undo.clicked.connect(self.pCanvas.box_undo)
        self.controlP.anno_tab.brush_reset.clicked.connect(
            self.pCanvas.brush_reset)
        self.controlP.anno_tab.brush_undo.clicked.connect(
            self.pCanvas.brush_undo)
        self.controlP.anno_tab.brush_size.valueChanged.connect(
            self.pCanvas.set_brush_size)

    def maskTabConnection(self):
        self.controlP.mask_tab.params[list].connect(
            self.pCanvas.set_mask_params)
        self.controlP.mask_tab.deleteIdx[int].connect(self.pCanvas.remove_mask)

    def loadDataConnection(self):
        self.controlP.loadBtn.data[np.ndarray].connect(self.pCanvas.set_data)


class MyMainWindow(qtw.QMainWindow):

    def __init__(self, nx: int = None, ny: int = None, clear_dim: bool = True):
        super().__init__()

        self.gstates = GlobalState(nx, ny)
        self.clear_dim = clear_dim
        self.initUI(nx, ny)

    def initUI(self, nx, ny):
        central_widget = qtw.QWidget()
        self.main_layout = qtw.QHBoxLayout()
        self.controlP = ControlP(self.gstates, nx, ny, self.clear_dim)
        self.main_layout.addWidget(self.controlP)
        self.plotCanvas = PlotCanvas(self, width=5, height=4)
        self.main_layout.addWidget(self.plotCanvas)
        self.centerControl = CentralController(self.controlP, self.plotCanvas)

        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)


def gui2d(nx: int = None, ny: int = None, clear_dim: bool = True):
    system = platform.system()
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

    app = qtw.QApplication(sys.argv)

    if system == 'Linux':
        font = QFont('Ubuntu')
        app.setFont(font)
    elif system == 'Windows':
        font = QFont('Segoe UI')
        app.setFont(font)

    main = MyMainWindow(nx, ny, clear_dim)
    main.show()
    sys.exit(app.exec_())
