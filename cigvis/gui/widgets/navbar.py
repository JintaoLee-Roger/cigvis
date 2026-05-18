"""Compact icon navbar used to switch drawer modules."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QButtonGroup, QFrame, QToolButton, QVBoxLayout


class NavBar(QFrame):
    """Narrow vertical tab bar with toggleable icon buttons."""

    idx_changed = Signal(int)

    def __init__(self, items: list, parent=None):
        super().__init__(parent)
        self.setFixedWidth(58)
        self.setObjectName("NavBar")

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(7, 10, 7, 10)
        self.layout.setSpacing(8)

        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(False)
        self.btn_group.idClicked.connect(self._on_btn_clicked)

        self._current_idx = -1
        self._items = items
        self._build_buttons()

    def _build_buttons(self) -> None:
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for idx, (name, icon_text) in enumerate(self._items):
            btn = QToolButton()
            btn.setText(icon_text)
            btn.setToolTip(name)
            btn.setCheckable(True)
            btn.setFixedSize(44, 44)

            self.layout.addWidget(btn)
            self.btn_group.addButton(btn, idx)

        self.layout.addStretch()

    def _on_btn_clicked(self, idx):
        sender = self.btn_group.button(idx)

        if idx == self._current_idx:
            self.btn_group.setExclusive(False)
            sender.setChecked(False)
            self._current_idx = -1
            self.idx_changed.emit(-1)
        else:
            self._current_idx = idx
            for btn in self.btn_group.buttons():
                if btn is not sender:
                    btn.setChecked(False)
            sender.setChecked(True)
            self.idx_changed.emit(idx)

    def clear_selection(self):
        """Programmatically clear current tab selection without emitting signals."""
        self.btn_group.setExclusive(False)
        for btn in self.btn_group.buttons():
            btn.setChecked(False)
        self._current_idx = -1

    def select_tab(self, idx: int) -> None:
        """Programmatically select a tab, equivalent to clicking it."""
        if idx < 0:
            self.clear_selection()
            self.idx_changed.emit(-1)
            return
        btn = self.btn_group.button(idx)
        if btn is None:
            return
        for b in self.btn_group.buttons():
            b.setChecked(False)
        btn.setChecked(True)
        self._current_idx = idx
        self.idx_changed.emit(idx)
