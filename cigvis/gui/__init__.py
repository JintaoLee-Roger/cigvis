"""Compatibility stubs for removed standalone cigvis GUI entry points.

The old standalone 2D/3D viewers have been removed. The only GUI kept inside
``cigvis`` is the lightweight shell used internally by
``cigvis.plot3D(..., gui=True)``.
"""

from __future__ import annotations


_REMOVED_MESSAGE = (
    "The standalone cigvis GUI has been removed. For existing VisPy nodes, "
    "use `cigvis.plot3D(..., gui=True)`. For SSH-friendly 2D slice viewing, "
    "install and use the `cigvis[sliceviewer]` extra."
)


Gui2dWindow = None

try:
    from .gui3d import Gui3dWindow, Plot3DGuiWindow, launch_plot3d_gui
except Exception:
    Gui3dWindow = None
    Plot3DGuiWindow = None
    launch_plot3d_gui = None


def gui2d(*_args, **_kwargs):
    """Removed standalone 2D GUI entry point."""
    raise RuntimeError(_REMOVED_MESSAGE)


def gui3d(*_args, **_kwargs):
    """Removed standalone 3D GUI entry point."""
    raise RuntimeError(_REMOVED_MESSAGE)


__all__ = [
    "gui2d",
    "gui3d",
    "Gui2dWindow",
    "Gui3dWindow",
    "Plot3DGuiWindow",
    "launch_plot3d_gui",
]
