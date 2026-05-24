# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
CIGVis - a tool for visualizing multidimensional geophysical data
==================================================================

**cigvis** is a tool for visualizing multidimensional geophysical data, 
developed by the 
`Computational Interpretation Group (CIG) <https://cig.ustc.edu.cn/main.htm>`_. 
Users can quickly visualize data with just a few lines of code.

cigvis can be used for various geophysical data visualizations, 
including 3D seismic data, overlays of seismic data with other 
information like labels, faults, RGT, horizon surfaces, well log 
trajectories, and well log curves, 3D geological bodies, 2D data, 
and 1D data, among others. Its GitHub repository can be found at 
`github.com/JintaoLee-Roger/cigvis <https://github.com/JintaoLee-Roger/cigvis>`_, 
and documentation is available at 
`https://cigvis.readthedocs.io/ <https://cigvis.readthedocs.io/>`.

cigvis leverages the power of underlying libraries such as 
`vispy <https://github.com/vispy/vispy>`_ for 3D visualization, 
`matplotlib <https://matplotlib.org/>`_ for 2D and 1D visualization, 
and `plotly <https://plotly.com/>`_ for Jupyter environments (work in 
progress). The 3D visualization component is heavily based on the code from 
`yunzhishi/seismic-canvas <https://github.com/yunzhishi/seismic-canvas>`_ 
and has been further developed upon this foundation.
"""


class ExceptionWrapper:
    """
    Copy from `trimesh.exceptions.ExceptionWrapper`

    Create a dummy object which will raise an exception when attributes
    are accessed (i.e. when used as a module) or when called (i.e.
    when used like a function)

    For soft dependencies we want to survive failing to import but
    we would like to raise an appropriate error when the functionality is
    actually requested so the user gets an easily debuggable message.
    """

    def __init__(self, e, custom=''):
        if custom:
            message = str(e) or e.__class__.__name__
            self.exception = type(e)(f"{message}\n\t{custom}", *e.args[1:])
        else:
            self.exception = e

    def __getattribute__(self, *args, **kwargs):
        if args[0] == "__class__":
            return None.__class__
        raise super().__getattribute__("exception")

    def __call__(self, *args, **kwargs):
        raise super().__getattribute__("exception")


def is_running_in_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        elif shell == 'TerminalInteractiveShell':
            return False
        else:
            return False
    except Exception:
        return False


import importlib.util


_has_vispy = importlib.util.find_spec("vispy") is not None

_VISPYPLOT_EXPORTS = [
    "create_slices",
    "add_mask",
    "create_overlay",
    "create_colorbar",
    "create_colorbar_from_nodes",
    "create_surfaces",
    "set_surface_color_by_slices_nodes",
    "create_bodies",
    "create_bodys",
    "create_line_logs",
    "create_Line_logs",
    "create_well_logs",
    "create_points",
    "create_point_cloud",
    "create_splats",
    "create_fault_skin",
    "create_arbitrary_line",
    "create_axis",
    "Plot3DView",
    "Plot3DSave",
    "Plot3DColorbar",
    "Plot3DGui",
    "plot3D",
    "run",
]

def _install_vispyplot_exports(module):
    for name in getattr(module, "__all__", _VISPYPLOT_EXPORTS):
        globals()[name] = getattr(module, name)


def _install_vispyplot_stub(exc):
    global vispyplot
    vispyplot = ExceptionWrapper(
        exc,
        "VisPy backend is optional. If you only need viserplot, use "
        "`from cigvis import viserplot` or `import cigvis.viserplot`. "
        "To use vispyplot, install the VisPy runtime dependencies "
        "(for example system fontconfig in minimal containers) or run "
        "`pip install \"cigvis[gui]\"` / `pip install \"cigvis[all]\"`."
    )
    for name in _VISPYPLOT_EXPORTS:
        globals()[name] = vispyplot


from .config import *
from . import io
from . import colormap
from . import meshs
_QT_GUI_BINDINGS = ("PySide6", "PyQt6", "PyQt5")
_has_qt_gui = any(importlib.util.find_spec(name) is not None for name in _QT_GUI_BINDINGS)

# GUI compatibility stubs are loaded lazily to avoid importing Qt unless needed.
# Standalone gui2d/gui3d have been removed; use plot3D(gui=True) for node inspection.
_lazy_modules = {}
if _has_vispy and _has_qt_gui:
    _lazy_modules['gui'] = 'cigvis.gui'


def __getattr__(name):
    if name in _lazy_modules:
        import importlib
        mod = importlib.import_module(_lazy_modules[name])
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'cigvis' has no attribute {name!r}")

if _has_vispy:
    try:
        from . import vispyplot
        _install_vispyplot_exports(vispyplot)
    except BaseException as E:
        _has_vispy = False
        _install_vispyplot_stub(E)
else:
    _install_vispyplot_stub(ImportError("vispy not found"))

try:
    from . import plotlyplot
except BaseException as E:
    plotlyplot = ExceptionWrapper(
        E,
        "run `pip install \"cigvis[plotly]\"` or run `pip install \"cigvis[all]\"` to install the dependencies"
    )

try:
    from . import viserplot
except BaseException as E:
    viserplot = ExceptionWrapper(
        E,
        "run `pip install \"cigvis[viser]\"` or run `pip install \"cigvis[all]\"` to install the dependencies"
    )

try:
    from . import sliceviewer
except BaseException as E:
    sliceviewer = ExceptionWrapper(
        E,
        "run `pip install \"cigvis[sliceviewer]\"` or `pip install panel plotly anywidget` to enable sliceviewer"
    )

from .mpl2dplot import *
from .mpl1dplot import *
from . import colors
from .mplstyle import load_theme
