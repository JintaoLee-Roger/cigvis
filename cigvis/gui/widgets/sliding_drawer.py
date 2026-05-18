"""Animated side drawer anchored to the navbar edge."""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtWidgets import QLabel, QHBoxLayout, QStackedWidget, QToolButton, QVBoxLayout, QWidget
from PySide6.QtCore import QPropertyAnimation, QEasingCurve, QPoint, Signal, Qt


class SlidingDrawer(QWidget):
    """Slide-in drawer that hosts stacked side-panel pages."""

    close_requested = Signal()

    def __init__(self, parent=None, width=300, animation_ms=250):
        super().__init__(parent)
        self.target_width = width
        self._titles: List[str] = []

        self.setObjectName("SlidingDrawer")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAutoFillBackground(True)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.header = QWidget(self)
        self.header.setObjectName("DrawerHeader")
        self.header.setAttribute(Qt.WA_StyledBackground, True)
        header_layout = QVBoxLayout(self.header)
        header_layout.setContentsMargins(14, 12, 10, 10)
        header_layout.setSpacing(6)

        self.title_label = QLabel("Controls", self.header)
        self.title_label.setObjectName("DrawerTitle")
        self.close_btn = QToolButton(self.header)
        self.close_btn.setObjectName("DrawerCloseButton")
        self.close_btn.setText("×")
        self.close_btn.setToolTip("Collapse sidebar")
        self.close_btn.clicked.connect(self.close_requested)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(8)
        title_row.addWidget(self.title_label, 1)
        title_row.addWidget(self.close_btn)
        header_layout.addLayout(title_row)

        self.main_layout.addWidget(self.header)

        self.stack = QStackedWidget(self)
        self.stack.setObjectName("DrawerStack")
        self.stack.setAttribute(Qt.WA_StyledBackground, True)
        self.main_layout.addWidget(self.stack, 1)

        self.anim = QPropertyAnimation(self, b"pos")
        self.anim.setDuration(animation_ms)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)

        self._pending_hide = False
        self.anim.finished.connect(self._on_anim_finished)

        self.hide()

    def add_module(self, widget: QWidget, title: Optional[str] = None):
        idx = self.stack.addWidget(widget)
        if title is not None:
            if idx < len(self._titles):
                self._titles[idx] = title
            else:
                self._titles.append(title)
        elif idx >= len(self._titles):
            self._titles.append("")
        if self.stack.count() == 1:
            self._update_title(0)

    def set_titles(self, titles: List[str]) -> None:
        self._titles = list(titles)
        self._update_title(self.stack.currentIndex())

    def set_content_widget(self, index: int):
        self.stack.setCurrentIndex(index)
        self._update_title(index)

    # alias used in mainwindow
    def set_page(self, index: int):
        self.stack.setCurrentIndex(index)
        self._update_title(index)

    def _update_title(self, index: int) -> None:
        if 0 <= index < len(self._titles) and self._titles[index]:
            self.title_label.setText(self._titles[index])
        else:
            self.title_label.setText("Controls")

    def _on_anim_finished(self):
        if self._pending_hide:
            self._pending_hide = False
            self.hide()

    def toggle(self, show: bool, anchor_x: int, parent_height: int):
        """Show or hide the drawer relative to the navbar right edge."""
        self.anim.stop()
        self._pending_hide = False

        self.resize(self.target_width, parent_height)

        if show:
            was_hidden = self.isHidden()
            self.show()
            self.raise_()

            end_pos = QPoint(anchor_x, 0)

            if was_hidden:
                self.move(-self.target_width, 0)

            self.anim.setStartValue(self.pos())
            self.anim.setEndValue(end_pos)
            self.anim.start()

        else:
            if self.isHidden():
                self.move(-self.target_width, 0)
                return

            end_pos = QPoint(-self.target_width, 0)
            self.anim.setStartValue(self.pos())
            self.anim.setEndValue(end_pos)
            self._pending_hide = True
            self.anim.start()
