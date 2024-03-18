from PyQt5.QtWidgets import QWidget
import numpy as np

import sys
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore, QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import cigvis
from cigvis import colormap
from .custom_widgets import *

CANVAS_SIZE = (800, 600)  # (width, height)

INT_validator = QtCore.QRegExp(r"^[1-9][0-9]*$")
FLOAT_validator = QtCore.QRegExp(r"[-+]?[0-9]*\.?[0-9]+")


class LoadBtn(qtw.QPushButton):
    dataLoaded = QtCore.pyqtSignal(np.ndarray)  # 用于通知数据加载完成的信号

    def __init__(self, parent=None):
        super(LoadBtn, self).__init__("Load File", parent)
        self.clicked.connect(self.loadData)

    def _is_base(self):
        if self.parent().loadRad.getCurrentSelection() == 'base':
            return True
        else:
            return False

    def _check_load_valid(self):
        flag = True
        if self.parent().dataLoadedFlag and self._is_base():
            qtw.QMessageBox.critical(
                self, "Warn",
                "Data already loaded. Please clear existing data first.")
            flag = False

        if not self.parent().dataLoadedFlag and not self._is_base():
            qtw.QMessageBox.critical(
                self, "Warn",
                "Base Image is empty. Please load base image first.")
            flag = False
        return flag

    def loadData(self, filePath: str = None):
        if not self._check_load_valid():
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
                self.parent().nx_input.setText(f'{nx}')
                self.parent().ny_input.setText(f'{ny}')
            else:
                nxc, nyc = self.parent().getNXNY()
                if nxc != nx or nyc != ny:
                    qtw.QMessageBox.critical(
                        self, "Warn",
                        f"Mask image's shape must be same as base image, but base is ({nxc}, {nyc}), mask is ({nx}, {ny})") # yapf: disable
                    return
        else:
            nx, ny = self.parent().getNXNY()
            if nx is None or ny is None:
                qtw.QMessageBox.critical(self, "Error",
                                         "Please enter values for nx, ny")
                return
            data = np.fromfile(filePath, np.float32).reshape(nx, ny)

        self.parent().dataLoadedFlag = True  # 标记数据已加载
        if self._is_base():
            self.parent().vmin_input.setText(f'{data.min():.2f}')
            self.parent().vmax_input.setText(f'{data.max():.2f}')
        else:
            # TODO:
            raise NotImplementedError("")

        self.dataLoaded.emit(data)  # 发送数据加载完成的信号


class ImageMixin:

    def set_data(self, data):
        self.data = data
        self.plot()

    def set_cmap(self, cmap):
        try:
            self.params['cmap'] = colormap.get_cmap_from_str(cmap)
            if self.baseim:
                self.baseim.set_cmap(self.params['cmap'])
                self.draw()
        except Exception as e:
            qtw.QMessageBox.critical(self, "Error", f"Error colormap: {e}")

    def set_interp(self, interp):
        self.params['interpolation'] = interp
        if self.baseim:
            self.baseim.set_interpolation(interp)
            self.draw()

    def set_vmin(self, vmin):
        vmin = float(vmin)
        self.params['vmin'] = vmin
        vmax = self.params['vmax'] if 'vmax' in self.params else -vmin
        clim = [vmin, vmax]
        if self.baseim:
            self.baseim.set(clim=clim)
            self.draw()

    def set_vmax(self, vmax):
        vmax = float(vmax)
        self.params['vmax'] = vmax
        vmin = self.params['vmin'] if 'vmin' in self.params else -vmax
        clim = [vmin, vmax]
        if self.baseim:
            self.baseim.set(clim=clim)
            self.draw()


class AnnotationMixin:

    def set_marker_mode(self, mode: int):
        self.marker_mode = mode

    def set_box_mode(self, mode: int):
        self.box_mode = mode

    def draw_marker(self, x, y):
        c = '#74ddd0' if self.marker_mode == 2 else '#f5c2cb'
        self.axes.plot(x, y, marker='o', color=c)
        self.draw()

    def on_mouse_press(self, event):
        if self.marker_mode >= 0 and event.inaxes:
            c = '#74ddd0' if self.marker_mode == 1 else '#f5c2cb'
            im = self.axes.plot(event.xdata, event.ydata, marker='o', color=c)
            self.marker_im.append(im[0])
            self.draw()
            self.marker_list.append(
                [event.xdata, event.ydata, self.marker_mode - 1])

        elif self.box_mode > 0 and event.inaxes:
            self.rectp.add_p0event(event)
            self.rect = Rectangle((event.xdata, event.ydata),
                                  0,
                                  0,
                                  fill=False,
                                  color='white')
            patch = self.axes.add_patch(self.rect)
            self.box_im.append(patch)

            self.draw()

    def on_mouse_move(self, event):
        if self.box_mode > 0 and self.rect is not None and event.inaxes:
            self.rectp.add_p1event(event)
            self.rect.set_width(self.rectp.x1 - self.rectp.x0)
            self.rect.set_height(self.rectp.y1 - self.rectp.y0)
            self.rect.set_xy((self.rectp.x0, self.rectp.y0))
            self.draw()

    def on_mouse_release(self, event):
        if self.box_mode > 0 and event.inaxes:
            self.box_list.append(RectP(*self.rectp.to_points()))
            self.rectp = RectP()
            self.rect = None

    def marker_undo(self):
        if len(self.marker_list) == 0:
            return
        p = self.marker_list.pop()
        im = self.marker_im.pop()
        im.remove()
        self.draw()

    def box_undo(self):
        if len(self.box_list) == 0:
            return
        p = self.box_list.pop()
        im = self.box_im.pop()
        im.remove()
        self.draw()


    def marker_reset(self):
        self.marker_list.clear()
        for im in self.marker_im:
            im.remove()
        self.marker_im.clear()
        self.draw()


    def box_reset(self):
        self.box_list.clear()
        for im in self.box_im:
            im.remove()
        self.box_im.clear()
        self.draw()

# class MaskImageMixin:
#     def set_mask_data(self, data):
#         self.mask_list.append(data)


class DraggableMixin:

    def enableDragging(self):
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        filePath = event.mimeData().urls()[0].toLocalFile()
        self.controlP.loadBtn.loadData(filePath)

    def handleFileDropped(self):
        raise NotImplementedError("Need Implemented in main class")


class AnnotationWidget(qtw.QWidget):
    pospointS = QtCore.pyqtSignal(int)
    boxS = QtCore.pyqtSignal(int)

    def __init__(self, grid_layout, parent=None):
        super(AnnotationWidget, self).__init__(parent)
        self.hover_pos = ToggleButton("+")
        self.hover_neg = ToggleButton("-")
        self.hover_reset = qtw.QPushButton("reset")
        self.hover_undo = qtw.QPushButton('undo')

        self.box_add = ToggleButton("+")
        self.box_reset = qtw.QPushButton("reset")
        self.box_undo = qtw.QPushButton('undo')

        self.hover_btn = ToggleButton("Hover")
        self.box_btn = ToggleButton("Box")

        grid_layout.addWidget(self.hover_btn, 0, 0, 1, 2)
        grid_layout.addWidget(self.hover_pos, 1, 0)
        grid_layout.addWidget(self.hover_neg, 1, 1)
        grid_layout.addWidget(self.hover_reset, 2, 0)
        grid_layout.addWidget(self.hover_undo, 2, 1)

        grid_layout.addWidget(self.box_btn, 0, 2, 1, 2)
        grid_layout.addWidget(self.box_add, 1, 2, 1, 2)
        grid_layout.addWidget(self.box_reset, 2, 2)
        grid_layout.addWidget(self.box_undo, 2, 3)
        self.connect_btn()

    def connect_btn(self):
        self.hover_btn.clicked.connect(self.active_marker)
        self.hover_pos.clicked.connect(self.active_marker)
        self.hover_neg.clicked.connect(self.active_marker)
        self.box_btn.clicked.connect(self.active_box)
        self.box_add.clicked.connect(self.active_box)

    def updateToggleState(self, toggledButton, checked):
        # 当一个按钮被切换到 ON，确保另一个按钮是 OFF
        if checked:
            if toggledButton == self.hover_btn:
                self.box_btn.setChecked(False)
                self.box_add.setChecked(False)
            elif toggledButton == self.box_btn:
                self.hover_btn.setChecked(False)
                self.hover_pos.setChecked(False)
                self.hover_neg.setChecked(False)
            elif toggledButton == self.hover_pos:
                self.hover_neg.setChecked(False)
                self.hover_btn.setChecked(True)
            elif toggledButton == self.hover_neg:
                self.hover_pos.setChecked(False)
                self.hover_btn.setChecked(True)
            elif toggledButton == self.box_add:
                self.box_btn.setChecked(True)
        else:
            if toggledButton == self.hover_btn:
                self.hover_pos.setChecked(False)
                self.hover_neg.setChecked(False)
            elif toggledButton == self.box_btn:
                self.box_add.setChecked(False)

    def active_marker(self):
        if not self.parent().dataLoadedFlag:
            self.pospointS.emit(-1)
            return

        if not self.hover_btn.isChecked():
            self.pospointS.emit(-1)
            return
        else:
            self.boxS.emit(-1)

        if self.hover_pos.isChecked():
            self.pospointS.emit(1)
        elif self.hover_neg.isChecked():
            self.pospointS.emit(0)

    def active_box(self):
        if not self.parent().dataLoadedFlag:
            self.boxS.emit(-1)
            return
        if not self.box_btn.isChecked():
            self.boxS.emit(-1)
            return
        else:
            self.pospointS.emit(-1)

        if self.box_add.isChecked():
            self.boxS.emit(1)

    def clear(self):
        self.box_btn.setChecked(False)
        self.hover_btn.setChecked(False)


class ControlP(qtw.QWidget):

    def __init__(self, nx=None, ny=None, clear_dim=True, parent=None):
        super().__init__(parent)

        self.clear_dim = clear_dim
        self.dataLoadedFlag = False
        self.loadType = 'base'

        layout = qtw.QVBoxLayout()

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
        self.addwidgets(
            row1_layout,
            [nx_label, self.nx_input, ny_label, self.ny_input],
        )

        row2_layout = qtw.QHBoxLayout()
        self.loadBtn = LoadBtn(self)
        self.loadRad = RadioButtonPanel(['base', 'mask'])
        self.addwidgets(row2_layout, [self.loadBtn, self.loadRad])

        rowbase_layout = qtw.QHBoxLayout()
        rowbase_layout.addStretch(1)
        base_label = qtw.QLabel('Base Image')
        self.addwidgets(rowbase_layout, [base_label])
        rowbase_layout.addStretch(1)

        # clim
        row3_layout = qtw.QHBoxLayout()
        vmin_label = qtw.QLabel('vmin:')
        self.vmin_input = qtw.QLineEdit()
        self.vmin_input.setValidator(
            QtGui.QRegExpValidator(FLOAT_validator, self))
        vmax_label = qtw.QLabel('vmax:')
        self.vmax_input = qtw.QLineEdit()
        self.vmax_input.setValidator(
            QtGui.QRegExpValidator(FLOAT_validator, self))
        self.addwidgets(
            row3_layout,
            [vmin_label, self.vmin_input, vmax_label, self.vmax_input])

        # colormap
        row4_layout = qtw.QHBoxLayout()
        colormap_label = qtw.QLabel('cmap:')
        self.colormap_combo = EditableComboBox()
        colormaps = [
            'gray', 'seismic', 'Petrel', 'stratum', 'od_seismic1',
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

        self.addwidgets(row4_layout, [
            colormap_label, self.colormap_combo, interp_label,
            self.interp_combo
        ])

        rowann_layout = qtw.QHBoxLayout()
        rowann_layout.addStretch(1)
        ann_label = qtw.QLabel('Annotation')
        self.addwidgets(rowann_layout, [ann_label])
        rowann_layout.addStretch(1)

        # annotation
        grid_layout = qtw.QGridLayout()
        self.annot = AnnotationWidget(grid_layout, parent=self)

        # clear and save
        rowl_layout = qtw.QHBoxLayout()
        self.save_btn = qtw.QPushButton('save')
        self.clear_btn = qtw.QPushButton('clear')
        self.addwidgets(rowl_layout, [self.save_btn, self.clear_btn])

        self.addlayout(layout, [
            row1_layout,
            row2_layout,
            rowbase_layout,
            row3_layout,
            row4_layout,
            rowann_layout,
            grid_layout,
            rowl_layout,
        ])

        layout.addStretch(1)
        self.setLayout(layout)
        self.setMaximumWidth(300)

    def addwidgets(self, layout, widgets):
        for widget in widgets:
            layout.addWidget(widget)

    def addlayout(self, layout, sublayouts):
        for sublayout in sublayouts:
            layout.addLayout(sublayout)

    def getNXNY(self):
        try:
            nx = int(self.nx_input.text())
            ny = int(self.ny_input.text())
            return nx, ny
        except ValueError:
            return None, None

    def clear(self):
        if self.clear_dim:
            self.nx_input.clear()
            self.ny_input.clear()
        self.vmin_input.clear()
        self.vmax_input.clear()
        self.colormap_combo.setCurrentText('gray')
        self.interp_combo.setCurrentText('bilinear')
        self.dataLoadedFlag = False
        self.annot.clear()


class PlotCanvas(FigureCanvas, DraggableMixin, ImageMixin, AnnotationMixin):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_axis_off()
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        super().__init__(self.fig)
        self.setParent(parent)
        self.controlP = self.parent().controlP
        self.enableDragging()
        self.mpl_connect('button_press_event', self.on_mouse_press)
        self.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.mpl_connect('button_release_event', self.on_mouse_release)
        self.init_states()

    def init_states(self):
        self.data = None
        self.params = {}
        self.baseim = None

        self.marker_mode = -1
        self.marker_list = []  # to save point number
        self.marker_im = []  # to save plt result

        self.box_mode = -1
        self.rect = None  # temp
        self.rectp = RectP()  # record
        self.box_list = []  # to save rectPs
        self.box_im = []  # to save plt result

    def plot(self):
        self.axes.clear()
        if self.data is None:
            self.axes.axis('off')
            self.baseim = None
        else:
            self.baseim = self.axes.imshow(self.data.T, **self.params)
        self.draw()

    def clear(self):
        self.data = None
        self.params = {}
        self.marker_reset()
        self.box_reset()
        self.plot()

    def save_fig(self):
        if not self.baseim:
            qtw.QMessageBox.critical(self, "Warn", "No Image")
            return

        fileName, _ = qtw.QFileDialog.getSaveFileName(
            self, "Save name", "",
            "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)")

        if fileName:
            self.fig.savefig(fileName,
                             bbox_inches='tight',
                             pad_inches=0.01,
                             dpi=300)


class CentralController(QtCore.QObject):

    def __init__(self, controlPanel, plotCanvas):
        self.controlP = controlPanel
        self.pCanvas = plotCanvas
        self.setupConnections()

    def setupConnections(self):
        self.loadDataConnection()
        self.controlP.colormap_combo.changed.connect(self.pCanvas.set_cmap)
        self.controlP.interp_combo.currentTextChanged.connect(
            self.pCanvas.set_interp)
        self.controlP.vmin_input.editingFinished.connect(
            lambda: self.pCanvas.set_vmin(self.controlP.vmin_input.text()))
        self.controlP.vmax_input.editingFinished.connect(
            lambda: self.pCanvas.set_vmax(self.controlP.vmax_input.text()))
        self.controlP.clear_btn.clicked.connect(self.pCanvas.clear)
        self.controlP.clear_btn.clicked.connect(self.controlP.clear)
        self.controlP.save_btn.clicked.connect(self.pCanvas.save_fig)

        # annotation
        self.controlP.annot.pospointS[int].connect(
            self.pCanvas.set_marker_mode)
        self.controlP.annot.boxS[int].connect(self.pCanvas.set_box_mode)
        self.controlP.annot.hover_reset.clicked.connect(
            self.pCanvas.marker_reset)
        self.controlP.annot.hover_undo.clicked.connect(
            self.pCanvas.marker_undo)
        self.controlP.annot.box_reset.clicked.connect(self.pCanvas.box_reset)
        self.controlP.annot.box_undo.clicked.connect(self.pCanvas.box_undo)

    def loadDataConnection(self):
        self.controlP.loadBtn.dataLoaded[np.ndarray].connect(
            lambda: self.pCanvas.set_vmin(self.controlP.vmin_input.text()))
        self.controlP.loadBtn.dataLoaded[np.ndarray].connect(
            lambda: self.pCanvas.set_vmax(self.controlP.vmax_input.text()))
        self.controlP.loadBtn.dataLoaded[np.ndarray].connect(
            lambda: self.pCanvas.set_cmap(self.controlP.colormap_combo.
                                          currentText()))
        self.controlP.loadBtn.dataLoaded[np.ndarray].connect(
            lambda: self.pCanvas.set_interp(self.controlP.interp_combo.
                                            currentText()))
        self.controlP.loadBtn.dataLoaded[np.ndarray].connect(
            self.pCanvas.set_data)


class MyMainWindow(qtw.QMainWindow):

    def __init__(self, nx=None, ny=None, clear_dim=True):
        super().__init__()

        self.clear_dim = clear_dim
        self.initUI(nx, ny)

    def initUI(self, nx, ny):
        central_widget = qtw.QWidget()
        self.main_layout = qtw.QHBoxLayout()
        self.controlP = ControlP(nx, ny, self.clear_dim)
        self.main_layout.addWidget(self.controlP)
        self.plotCanvas = PlotCanvas(self, width=5, height=4)
        self.main_layout.addWidget(self.plotCanvas)
        self.centerControl = CentralController(self.controlP, self.plotCanvas)

        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)


def gui2d(nx=None, ny=None, clear_dim=True):
    app = qtw.QApplication(sys.argv)
    main = MyMainWindow(nx, ny, clear_dim)
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    main = MyMainWindow()
    main.show()
    sys.exit(app.exec_())
