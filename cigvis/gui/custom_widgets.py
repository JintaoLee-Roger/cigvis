# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.

"""
Some custom widgets
"""
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore 

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