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

_GUI3D_EXPORTS = {
    "Gui3dWindow",
    "Plot3DGuiWindow",
    "launch_plot3d_gui",
}


def _load_gui3d_export(name: str):
    try:
        from . import gui3d as _gui3d
    except Exception:
        value = None
    else:
        value = getattr(_gui3d, name, None)
    globals()[name] = value
    return value


def gui2d(*_args, **_kwargs):
    """Removed standalone 2D GUI entry point."""
    raise RuntimeError(_REMOVED_MESSAGE)


def gui3d(*_args, **_kwargs):
    """Removed standalone 3D GUI entry point."""
    raise RuntimeError(_REMOVED_MESSAGE)


def __getattr__(name: str):
    if name in _GUI3D_EXPORTS:
        return _load_gui3d_export(name)
    raise AttributeError(f"module 'cigvis.gui' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "gui2d",
    "gui3d",
    "Gui2dWindow",
    "Gui3dWindow",
    "Plot3DGuiWindow",
    "launch_plot3d_gui",
]
