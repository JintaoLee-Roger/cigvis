"""
3D Viewer main window — modern collapsible sidebar layout.

Layout:
  [NavBar 50px] [SlidingDrawer 290px overlay] [3D Vispy Canvas flex]

NavBar tabs:
  0 📂  Load
  1 🎨  Display
  2 📷  Camera
  3 📐  Slices
  4 🗂   Overlays
  5 🤖  SAM  (optional)
"""

from __future__ import annotations

import sys
import platform
from typing import Any, Dict, Optional

import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout,
    QScrollArea, QStatusBar, QVBoxLayout,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut, QFont
from pathlib import Path

from cigvis.gui.widgets.navbar import NavBar
from cigvis.gui.widgets.sliding_drawer import SlidingDrawer
from .sidebar import (
    LoadPanel3D, DisplayPanel3D, CameraPanel, SlicesPanel,
    OverlaysPanel3D, SamPanel,
)
from .plot_canvas import PlotCanvas3D


def _load_stylesheet(theme: str = 'light') -> str:
    name = 'dark.qss' if theme == 'dark' else 'light.qss'
    qss_path = Path(__file__).parent.parent / 'styles' / name
    if qss_path.exists():
        return qss_path.read_text()
    return ""


def _drawer_page(*widgets: QWidget) -> QWidget:
    """Build one scrollable drawer page from multiple compact panels."""
    page = QWidget()
    page.setObjectName("DrawerPage")
    page.setAttribute(Qt.WA_StyledBackground, True)
    page.setAutoFillBackground(True)
    layout = QVBoxLayout(page)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(10)
    for widget in widgets:
        layout.addWidget(widget)
    layout.addStretch(1)
    return page


def _has_mask_overlays(nodes) -> bool:
    if nodes is None:
        return False
    if isinstance(nodes, dict):
        return any(_has_mask_overlays(value) for value in nodes.values())
    if isinstance(nodes, (list, tuple)):
        return any(_has_mask_overlays(item) for item in nodes)
    images = getattr(nodes, 'overlaid_images', None)
    return bool(images and len(images) > 1)


# ---------------------------------------------------------------------------
# SAM controller (wires SamPanel ↔ PlotCanvas3D)
# ---------------------------------------------------------------------------

class SamController:
    """
    Connects the SamPanel UI to the SamLikeVolumeApp logic.
    Only instantiated when SAM mode is enabled.
    """

    def __init__(self, panel: SamPanel, canvas: PlotCanvas3D,
                 decode_fn: Optional[callable] = None) -> None:
        self._panel = panel
        self._canvas = canvas
        self._sam_app = None
        self._decode_fn = decode_fn

    def activate(self) -> None:
        if self._canvas._vol is None:
            return
        try:
            from cigvis.gui.gui3d.sam_controller import SamLikeController
            self._sam_app = SamLikeController(
                vol=self._canvas._vol,
                canvas=self._canvas.canvas,
                decode_fn=self._decode_fn,
                on_prompt_count=self._panel.set_prompt_count,
            )
        except Exception as e:
            print(f"[SAM] Failed to activate: {e}")

    def deactivate(self) -> None:
        if self._sam_app and hasattr(self._sam_app, 'close'):
            self._sam_app.close()
        self._sam_app = None

    def run(self) -> None:
        if self._sam_app and hasattr(self._sam_app, 'submit_decode'):
            if self._sam_app.prompt_xyz is not None:
                self._sam_app.submit_decode(self._sam_app.prompt_xyz)

    def undo(self) -> None:
        if self._sam_app and hasattr(self._sam_app, '_undo_last_prompt'):
            self._sam_app._undo_last_prompt()

    def clear(self) -> None:
        if self._sam_app and hasattr(self._sam_app, '_clear_prompts'):
            self._sam_app._clear_prompts()
        self._panel.set_prompt_count(0)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class Gui3dWindow(QMainWindow):
    """
    Modern 3D seismic viewer with animated collapsible sidebar.

    Parameters
    ----------
    nx, ny, nz : int, optional
        Pre-set dimensions.
    clear_dim : bool
        Clear dimensions on data clear.
    data : ndarray, optional
        Pre-load data on startup.
    decode_fn : callable, optional
        SAM-like inference function. If provided, enables the SAM tab.
    """

    def __init__(
        self,
        nx: Optional[int] = None,
        ny: Optional[int] = None,
        nz: Optional[int] = None,
        clear_dim: bool = True,
        data: Optional[np.ndarray] = None,
        decode_fn: Optional[callable] = None,
        theme: str = 'light',
        nodes: Optional[list] = None,
        grid: Optional[tuple] = None,
        share: bool = False,
        canvas_kwargs: Optional[Dict[str, Any]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._clear_dim = clear_dim
        self._decode_fn = decode_fn
        self._plot_mode = nodes is not None
        self._plot_has_overlays = self._plot_mode and _has_mask_overlays(nodes)
        self._sam_ctrl: Optional[SamController] = None
        self._last_open_tab_idx: int = 0
        self._sync_timer: Optional[QTimer] = None
        self.setWindowTitle("CigVis 3D Viewer")
        self.resize(1150, 800)

        self.setStyleSheet(_load_stylesheet(theme))

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        if self._plot_mode:
            nav_items = [
                ("Visuals", "🎨"),
                ("View", "📷"),
            ]
            if self._plot_has_overlays:
                nav_items.append(("Layers", "🗂"))
        else:
            nav_items = [
                ("Load", "📂"),
                ("Visuals", "🎨"),
                ("View", "📷"),
                ("Layers", "🗂"),
            ]
            if decode_fn is not None:
                nav_items.append(("SAM", "🤖"))

        self.navbar = NavBar(nav_items)
        root.addWidget(self.navbar)

        # 3D canvas
        self.canvas = PlotCanvas3D(
            self,
            visual_nodes=nodes,
            grid=grid,
            share=share,
            canvas_kwargs=canvas_kwargs,
        )
        root.addWidget(self.canvas, 1)

        # SlidingDrawer
        drawer_width = 270 if self._plot_mode else 330
        self.drawer = SlidingDrawer(central, width=drawer_width)
        self.drawer.set_titles([name for name, _icon in nav_items])
        self.drawer.close_requested.connect(self._collapse_drawer)

        # Build panels
        self._display_panel = DisplayPanel3D(compact=self._plot_mode)
        self._camera_panel = CameraPanel(compact=self._plot_mode)
        self._slices_panel = SlicesPanel(compact=self._plot_mode)

        self._load_panel = None
        self._overlays_panel = None
        self._sam_panel = None

        if self._plot_mode:
            self._overlays_panel = OverlaysPanel3D() if self._plot_has_overlays else None
            pages = [
                _drawer_page(self._display_panel),
                _drawer_page(self._camera_panel, self._slices_panel),
            ]
            if self._overlays_panel is not None:
                self._overlays_panel.set_mask_items(self.canvas.get_mask_display_params())
                pages.append(_drawer_page(self._overlays_panel))
            limits = self.canvas.get_slice_limits()
            if limits:
                self._slices_panel.set_axis_limits(limits)
            positions = self.canvas.get_slice_positions()
            if positions:
                self._slices_panel.set_positions(positions)
        else:
            self._load_panel = LoadPanel3D()
            self._overlays_panel = OverlaysPanel3D()
            self._sam_panel = SamPanel() if decode_fn is not None else None
            pages = [
                _drawer_page(self._load_panel),
                _drawer_page(self._display_panel),
                _drawer_page(self._camera_panel, self._slices_panel),
                _drawer_page(self._overlays_panel),
            ]
            if self._sam_panel:
                pages.append(_drawer_page(self._sam_panel))

        for p in pages:
            scroll = QScrollArea()
            scroll.setObjectName("DrawerScroll")
            scroll.setAttribute(Qt.WA_StyledBackground, True)
            scroll.setAutoFillBackground(True)
            scroll.viewport().setObjectName("DrawerViewport")
            scroll.viewport().setAttribute(Qt.WA_StyledBackground, True)
            scroll.viewport().setAutoFillBackground(True)
            scroll.setWidgetResizable(True)
            scroll.setWidget(p)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.drawer.add_module(scroll)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Wire up
        self.navbar.idx_changed.connect(self._on_navbar)
        self._connect_panels()

        # Pre-populate dims
        if self._load_panel is not None and nx:
            self._load_panel.nx_edit.set_value(str(nx))
            self._load_panel._nx = nx
        if self._load_panel is not None and ny:
            self._load_panel.ny_edit.set_value(str(ny))
            self._load_panel._ny = ny
        if self._load_panel is not None and nz:
            self._load_panel.nz_edit.set_value(str(nz))
            self._load_panel._nz = nz

        if self._load_panel is not None and data is not None:
            self._load_panel.sent_data(data)

        self._init_shortcuts()
        if self._plot_mode:
            self._sync_controls_from_canvas(force=True)

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
            # _on_navbar is triggered via navbar.idx_changed signal

    def _collapse_drawer(self) -> None:
        self.navbar.clear_selection()
        self._on_navbar(-1)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self.drawer.isHidden():
            h = self.centralWidget().height()
            self.drawer.resize(self.drawer.target_width, h)
            self.drawer.move(self.navbar.width(), 0)

    def _on_navbar(self, idx: int) -> None:
        navbar_w = self.navbar.width()
        h = self.centralWidget().height()
        if idx < 0:
            self.drawer.toggle(False, navbar_w, h)
        else:
            self._last_open_tab_idx = idx
            self.drawer.set_page(idx)
            self._sync_controls_from_canvas(force=(idx == 0))
            self.drawer.toggle(True, navbar_w, h)

    def _sync_controls_from_canvas(self, force: bool = False) -> None:
        if force and hasattr(self.canvas, 'get_base_display_params'):
            params = self.canvas.get_base_display_params()
            cmap = params.get('cmap')
            if not isinstance(cmap, str):
                cmap = None
            self._display_panel.set_params(
                cmap=cmap,
                clim=params.get('clim'),
                interpolation=params.get('interpolation'),
            )

        camera_params = self.canvas.get_camera_params()
        if camera_params:
            self._camera_panel.update_from_params(camera_params)
            if len(camera_params) > 3 and isinstance(camera_params[3], dict):
                self._slices_panel.set_positions(camera_params[3])
        else:
            positions = self.canvas.get_slice_positions()
            if positions:
                self._slices_panel.set_positions(positions)

    def _connect_panels(self) -> None:
        lp = self._load_panel
        dp = self._display_panel
        cp = self._camera_panel
        sp = self._slices_panel
        op = self._overlays_panel
        cv = self.canvas

        # Load → canvas / other panels
        if lp is not None and op is not None:
            lp.base_loaded.connect(self._on_base_data)
            lp.mask_loaded.connect(cv.add_mask)
            lp.horz_loaded.connect(cv.add_horizon)
            lp.vmin_ready.connect(dp.set_vmin)
            lp.vmax_ready.connect(dp.set_vmax)
            lp.mask_item_ready.connect(op.add_mask_item)
            lp.horz_item_ready.connect(op.add_horz_item)
            lp.shape_ready.connect(sp.set_limits)
            lp.clear_btn.clicked.connect(self._clear_all)

        # Display → canvas
        dp.cmap_changed.connect(cv.set_cmap)
        dp.vmin_changed.connect(cv.set_vmin)
        dp.vmax_changed.connect(cv.set_vmax)
        dp.interp_changed.connect(cv.set_interp)

        # Camera → canvas
        cp.azimuth_changed.connect(cv.set_azimuth)
        cp.elevation_changed.connect(cv.set_elevation)
        cp.fov_changed.connect(cv.set_fov)
        cp.aspectx_changed.connect(cv.set_aspectx)
        cp.aspecty_changed.connect(cv.set_aspecty)
        cp.aspectz_changed.connect(cv.set_aspectz)
        cp.update_requested.connect(
            lambda: self._sync_controls_from_canvas(force=True))
        cp.live_sync_toggled.connect(self._set_live_sync)

        # Slices → canvas
        sp.xpos_changed.connect(cv.set_xpos)
        sp.ypos_changed.connect(cv.set_ypos)
        sp.zpos_changed.connect(cv.set_zpos)

        # Overlays → canvas
        if op is not None:
            op.mask_params_changed.connect(cv.set_mask_params)
            op.mask_deleted.connect(cv.remove_mask)
            op.horz_params_changed.connect(cv.set_horz_params)
            op.horz_deleted.connect(cv.remove_horizon)

        # SAM
        if self._sam_panel:
            self._sam_panel.sam_toggled.connect(self._on_sam_toggled)
            self._sam_panel.run_requested.connect(self._on_sam_run)
            self._sam_panel.undo_requested.connect(self._on_sam_undo)
            self._sam_panel.clear_requested.connect(self._on_sam_clear)

    def _set_live_sync(self, enabled: bool) -> None:
        if self._sync_timer is None:
            self._sync_timer = QTimer(self)
            self._sync_timer.setInterval(350)
            self._sync_timer.timeout.connect(self._sync_controls_from_canvas)

        if enabled:
            self._sync_controls_from_canvas(force=True)
            self._sync_timer.start()
        else:
            self._sync_timer.stop()

    def _on_base_data(self, data: np.ndarray) -> None:
        self.canvas.set_base_data(data)
        # Update slice limits after shape is known
        shape = self.canvas._vol.shape if self.canvas._vol else None
        if shape:
            self._slices_panel.set_limits(*shape)
            # Default slice positions to center
            self._slices_panel.xpos.setValue(shape[0] // 2)
            self._slices_panel.ypos.setValue(shape[1] // 2)
            self._slices_panel.zpos.setValue(shape[2] // 2)
        # Enable SAM if available
        if self._sam_panel:
            self._sam_panel.enable_sam()

    def _clear_all(self) -> None:
        if self._sam_ctrl:
            self._sam_ctrl.deactivate()
            self._sam_ctrl = None
        if self._sync_timer is not None:
            self._sync_timer.stop()
        self.canvas.clear()
        if self._load_panel is not None:
            self._load_panel.clear(clear_dims=self._clear_dim)
        self._display_panel.clear()
        self._camera_panel.clear()
        self._slices_panel.clear()
        if self._overlays_panel is not None:
            self._overlays_panel.clear()
        if self._sam_panel:
            self._sam_panel.setEnabled(False)
            self._sam_panel.set_prompt_count(0)

    def _on_sam_toggled(self, enabled: bool) -> None:
        if enabled:
            self._sam_ctrl = SamController(
                self._sam_panel, self.canvas, self._decode_fn)
            self._sam_ctrl.activate()
        else:
            if self._sam_ctrl:
                self._sam_ctrl.deactivate()
                self._sam_ctrl = None

    def _on_sam_run(self) -> None:
        if self._sam_ctrl:
            self._sam_ctrl.run()

    def _on_sam_undo(self) -> None:
        if self._sam_ctrl:
            self._sam_ctrl.undo()

    def _on_sam_clear(self) -> None:
        if self._sam_ctrl:
            self._sam_ctrl.clear()

    def on_file_dropped(self, path: str) -> None:
        if self._load_panel is not None:
            self._load_panel.load_file(path, check=False)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            if self.is_drawer_open:
                self._collapse_drawer()
            else:
                self.close()
            return
        super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# plot3D entry point
# ---------------------------------------------------------------------------

_REMOVED_MESSAGE = (
    "The standalone cigvis 3D GUI has been removed. For existing VisPy nodes, "
    "use `cigvis.plot3D(..., gui=True)`."
)


Plot3DGuiWindow = Gui3dWindow


def _configure_font(qt_app: QApplication) -> None:
    system = platform.system()
    if system == 'Linux':
        qt_app.setFont(QFont('Ubuntu'))
    elif system == 'Windows':
        qt_app.setFont(QFont('Segoe UI'))


def launch_plot3d_gui(
    *,
    nodes: Optional[list] = None,
    grid: Optional[tuple] = None,
    share: bool = False,
    theme: str = 'dark',
    canvas_kwargs: Optional[Dict[str, Any]] = None,
    run_app: bool = True,
) -> Plot3DGuiWindow:
    """Launch the retained ``plot3D(gui=True)`` PySide6 shell."""
    from vispy.app import use_app
    app_vispy = use_app("pyside6")
    app_vispy.create()

    qt_app = QApplication.instance() or QApplication(sys.argv)
    _configure_font(qt_app)

    win = Gui3dWindow(
        theme=theme,
        nodes=nodes,
        grid=grid,
        share=share,
        canvas_kwargs=canvas_kwargs,
    )
    win.show()
    if run_app:
        app_vispy.run()
    return win


def gui3d(*_args, **_kwargs):
    """Removed standalone 3D GUI entry point."""
    raise RuntimeError(_REMOVED_MESSAGE)
