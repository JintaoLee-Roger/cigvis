"""
2D Viewer main window — modern layout with collapsible sidebar.

Layout:
  [NavBar 50px] [SlidingDrawer 290px overlay] [PlotCanvas flex]

NavBar tabs:
  0 📂  Load
  1 🎨  Display
  2 📋  Slices   (only active when 3D data loaded)
  3 ✏️   Annotate
  4 🗂️   Overlays
"""

from __future__ import annotations

import sys
import platform
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QScrollArea,
    QSizePolicy, QStatusBar, QVBoxLayout,
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QKeySequence, QShortcut

from cigvis.gui.widgets.navbar import NavBar
from cigvis.gui.widgets.sliding_drawer import SlidingDrawer
from .sidebar import LoadPanel, DisplayPanel, SliceNavPanel, AnnotationPanel, OverlaysPanel
from .plot_canvas import PlotCanvas


def _load_stylesheet(theme: str = 'light') -> str:
    name = 'dark.qss' if theme == 'dark' else 'light.qss'
    qss_path = Path(__file__).parent.parent / 'styles' / name
    if qss_path.exists():
        return qss_path.read_text()
    return ""


def _drawer_page(*widgets: QWidget) -> QWidget:
    page = QWidget()
    page.setObjectName("DrawerPage")
    layout = QVBoxLayout(page)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(10)
    for widget in widgets:
        layout.addWidget(widget)
    layout.addStretch(1)
    return page


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class Gui2dWindow(QMainWindow):
    """
    Modern 2D viewer with animated collapsible sidebar.

    Parameters
    ----------
    nx, ny : int, optional
        Pre-set dimensions for binary file loading.
    clear_dim : bool
        Whether to clear dimensions on data clear (default True).
    data : ndarray, optional
        Pre-load data on startup.
    """

    def __init__(
        self,
        nx: Optional[int] = None,
        ny: Optional[int] = None,
        clear_dim: bool = True,
        data: Optional[np.ndarray] = None,
        theme: str = 'light',
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._clear_dim = clear_dim
        self._last_open_tab_idx: int = 0
        self.setWindowTitle("CigVis 2D Viewer")
        self.resize(1050, 750)

        # Apply stylesheet
        self.setStyleSheet(_load_stylesheet(theme))

        # Central widget: navbar + canvas (drawer overlaid)
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # --- NavBar ---
        nav_items = [
            ("Load", "📂"),
            ("Visuals", "🎨"),
            ("Tools", "✏"),
        ]
        self.navbar = NavBar(nav_items)
        root_layout.addWidget(self.navbar)

        # --- Canvas ---
        self.canvas = PlotCanvas(self)
        root_layout.addWidget(self.canvas, 1)

        # --- SlidingDrawer (overlaid on canvas) ---
        self.drawer = SlidingDrawer(central, width=320)
        self.drawer.set_titles([name for name, _icon in nav_items])
        self.drawer.close_requested.connect(self._collapse_drawer)

        # Build drawer pages
        self._load_panel = LoadPanel()
        self._display_panel = DisplayPanel()
        self._slice_panel = SliceNavPanel()
        self._anno_panel = AnnotationPanel()
        self._overlay_panel = OverlaysPanel()

        pages = [
            _drawer_page(self._load_panel),
            _drawer_page(self._display_panel, self._overlay_panel),
            _drawer_page(self._slice_panel, self._anno_panel),
        ]
        for panel in pages:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(panel)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.drawer.add_module(scroll)

        # --- Status bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.canvas.set_status_callback(self.status_bar.showMessage)

        # --- Connections ---
        self.navbar.idx_changed.connect(self._on_navbar)
        self._connect_panels()

        # Pre-load data if provided
        if nx is not None or ny is not None:
            if nx:
                self._load_panel.nx_edit.set_value(str(nx))
                self._load_panel._nx = nx
            if ny:
                self._load_panel.ny_edit.set_value(str(ny))
                self._load_panel._ny = ny

        if data is not None:
            self._on_base_data(data)

        self._init_shortcuts()

    @property
    def is_drawer_open(self) -> bool:
        return not self.drawer.isHidden()

    def _init_shortcuts(self) -> None:
        self._sc_collapse = QShortcut(QKeySequence("Esc"), self)
        self._sc_collapse.setContext(Qt.WindowShortcut)
        self._sc_collapse.activated.connect(self._on_shortcut_collapse_ui)

    def _on_shortcut_collapse_ui(self) -> None:
        if self.is_drawer_open:
            self._collapse_drawer()
        else:
            self.navbar.select_tab(self._last_open_tab_idx)

    def _collapse_drawer(self) -> None:
        self.navbar.clear_selection()
        self._on_navbar(-1)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_drawer()

    def _reposition_drawer(self) -> None:
        navbar_w = self.navbar.width()
        h = self.centralWidget().height()
        if not self.drawer.isHidden():
            self.drawer.resize(self.drawer.target_width, h)
            self.drawer.move(navbar_w, 0)

    def _on_navbar(self, idx: int) -> None:
        navbar_w = self.navbar.width()
        h = self.centralWidget().height()
        if idx < 0:
            self.drawer.toggle(False, navbar_w, h)
        else:
            self._last_open_tab_idx = idx
            self.drawer.set_page(idx)
            self.drawer.toggle(True, navbar_w, h)

    def _connect_panels(self) -> None:
        lp = self._load_panel
        dp = self._display_panel
        sp = self._slice_panel
        ap = self._anno_panel
        op = self._overlay_panel
        cv = self.canvas

        # Load panel → canvas
        lp.base_loaded.connect(self._on_base_data)
        lp.mask_loaded.connect(cv.add_mask)
        lp.vmin_ready.connect(dp.set_vmin)
        lp.vmax_ready.connect(dp.set_vmax)
        lp.mask_item_ready.connect(op.add_item)
        lp.clear_btn.clicked.connect(self._clear_all)
        lp.save_btn.clicked.connect(cv.save_fig)

        # Display panel → canvas
        dp.cmap_changed.connect(cv.set_cmap)
        dp.vmin_changed.connect(cv.set_vmin)
        dp.vmax_changed.connect(cv.set_vmax)
        dp.interp_changed.connect(cv.set_interp)

        # Slice navigation → canvas
        sp.slice_changed.connect(lambda ax, idx: cv.set_slice(ax, idx))

        # Annotation panel → canvas
        ap.marker_mode_changed.connect(cv.set_marker_mode)
        ap.box_mode_changed.connect(cv.set_box_mode)
        ap.brush_mode_changed.connect(cv.set_brush_mode)
        ap.brush_size_changed.connect(cv.set_brush_size)
        ap.marker_reset.connect(cv.do_marker_reset)
        ap.marker_undo.connect(cv.do_marker_undo)
        ap.box_reset.connect(cv.do_box_reset)
        ap.box_undo.connect(cv.do_box_undo)
        ap.brush_reset.connect(cv.do_brush_reset)
        ap.brush_undo.connect(cv.do_brush_undo)

        # Overlay panel → canvas
        op.params_changed.connect(cv.set_mask_params)
        op.item_deleted.connect(cv.remove_mask)

    def _on_base_data(self, data: np.ndarray) -> None:
        self.canvas.set_base_data(data)
        if data.ndim == 3:
            self._slice_panel.set_shape(data.shape)
        else:
            self._slice_panel.setEnabled(False)

    def _clear_all(self) -> None:
        self.canvas.clear()
        self._load_panel.clear(clear_dims=self._clear_dim)
        self._display_panel.clear()
        self._anno_panel.clear()
        self._overlay_panel.clear()
        self._slice_panel.setEnabled(False)

    def on_file_dropped(self, path: str) -> None:
        self._load_panel.load_file(path, check=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def gui2d(
    nx: Optional[int] = None,
    ny: Optional[int] = None,
    clear_dim: bool = True,
    data: Optional[np.ndarray] = None,
    theme: str = 'light',
) -> None:
    """
    Launch the 2D viewer.

    Parameters
    ----------
    theme : str
        UI theme, ``'light'`` (default) or ``'dark'``.
    """
    from PySide6.QtGui import QFont
    app = QApplication.instance() or QApplication(sys.argv)

    system = platform.system()
    if system == 'Linux':
        app.setFont(QFont('Ubuntu'))
    elif system == 'Windows':
        app.setFont(QFont('Segoe UI'))

    win = Gui2dWindow(nx=nx, ny=ny, clear_dim=clear_dim, data=data, theme=theme)
    win.show()
    sys.exit(app.exec())
