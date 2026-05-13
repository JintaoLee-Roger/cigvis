"""Collapsible section widget for dense side-panel layouts."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QPushButton, QSizePolicy, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """Toggle button + hideable body container."""

    def __init__(
        self,
        title: str,
        *,
        collapsed: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._title = title

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 4)
        root.setSpacing(2)

        self._btn = QPushButton()
        self._btn.setObjectName("SectionToggle")
        self._btn.setCheckable(True)
        self._btn.setChecked(not collapsed)
        self._btn.setFlat(True)
        self._btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._btn.clicked.connect(self._on_toggle)
        root.addWidget(self._btn)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(4, 0, 4, 0)
        self._body_layout.setSpacing(3)
        self._body.setVisible(not collapsed)
        root.addWidget(self._body)

        self._refresh_label()

    def add(self, child: QWidget) -> "CollapsibleSection":
        self._body_layout.addWidget(child)
        return self

    def add_layout(self, layout) -> "CollapsibleSection":
        self._body_layout.addLayout(layout)
        return self

    def _on_toggle(self, checked: bool) -> None:
        self._body.setVisible(checked)
        self._refresh_label()

    def _refresh_label(self) -> None:
        marker = "▾" if self._btn.isChecked() else "▸"
        self._btn.setText(f"  {marker}  {self._title}")
