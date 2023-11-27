# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.

import typing
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui, QtCore
from .custom_widgets import *

INT_validator = QtCore.QRegExp(r"^[1-9][0-9]*$")
FLOAT_validator = QtCore.QRegExp(r"[-+]?[0-9]*\.?[0-9]+")


class RowWidget(qtw.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        pass

    def addwidgets(self, layout, widgets):
        for widget in widgets:
            layout.addWidget(widget)

    def clear(self):
        raise NotImplementedError("Not implementated")


class DimsWidget(RowWidget):

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

    def set_dims(self, nx, ny, nz):
        self.nx_input.setText(f'{nx}')
        self.ny_input.setText(f'{ny}')
        self.nz_input.setText(f'{nz}')

    def clear(self):
        if self.clear_dim:
            self.nx_input.clear()
            self.ny_input.clear()
            self.nz_input.clear()


class ClimWidget(RowWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

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

    def clear(self):
        self.vmin_input.clear()
        self.vmax_input.clear()

    def set(self, vmin, vmax):
        self.vmin_input.setText(f'{vmin}')
        self.vmax_input.setText(f'{vmax}')


class CameraWidget(RowWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

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

    def clear(self):
        self.azimuth_input.setValue(50)
        self.elevation_input.setValue(50)
        self.fov_input.setValue(30)


class PosWidget(RowWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

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

    def clear(self):
        self.xpos.setValue(0)
        self.ypos.setValue(0)
        self.zpos.setValue(0)


class AspectWidget(RowWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

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

    def clear(self):
        self.aspx.setValue(1)
        self.aspy.setValue(1)
        self.aspz.setValue(1)