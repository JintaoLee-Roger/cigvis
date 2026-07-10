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

import os
import sys


def _suppress_qt_windows_dpi_warning():
    if sys.platform != 'win32':
        return
    key = 'QT_LOGGING_RULES'
    rule = 'qt.qpa.window.warning=false'
    existing = os.environ.get(key, '')
    if 'qt.qpa.window.warning' in existing:
        return
    os.environ[key] = f'{existing};{rule}' if existing else rule


_suppress_qt_windows_dpi_warning()


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


import importlib
import importlib.util

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

_MPL2D_EXPORTS = [
    "fg_image_args",
    "line_args",
    "marker_args",
    "annotate_args",
    "discrete_cbar",
    "plot2d",
    "flattened_view",
]

_MPL1D_EXPORTS = [
    "plot1d",
    "plot_multi_traces",
    "plot_signal_compare",
    "plot1_with_fill",
]

_CONFIG_EXPORTS = [
    "LINE_FIRST",
    "X_REVERSED",
    "Y_REVERSED",
    "Z_REVERSED",
    "is_line_first",
    "set_order",
    "is_x_reversed",
    "set_x_reversed",
    "is_y_reversed",
    "set_y_reversed",
    "is_z_reversed",
    "set_z_reversed",
    "is_axis_reversed",
    "set_axis_reversed",
]

_VISPYPLOT_ERROR = (
    "VisPy backend is optional. If you only need viserplot, use "
    "`from cigvis import viserplot` or `import cigvis.viserplot`. "
    "To use vispyplot, install the VisPy runtime dependencies "
    "(for example system fontconfig in minimal containers) or run "
    "`pip install \"cigvis[gui]\"` / `pip install \"cigvis[all]\"`."
)

_LAZY_MODULES = {
    "io": ("cigvis.io", None),
    "colormap": ("cigvis.colormap", None),
    "meshs": ("cigvis.meshs", None),
    "colors": ("cigvis.colors", None),
    "mplstyle": ("cigvis.mplstyle", None),
    "mpl2dplot": ("cigvis.mpl2dplot", None),
    "mpl1dplot": ("cigvis.mpl1dplot", None),
    "gui": ("cigvis.gui", None),
    "vispyplot": ("cigvis.vispyplot", _VISPYPLOT_ERROR),
    "plotlyplot": (
        "cigvis.plotlyplot",
        "run `pip install \"cigvis[plotly]\"` or run `pip install \"cigvis[all]\"` to install the dependencies",
    ),
    "viserplot": (
        "cigvis.viserplot",
        "run `pip install \"cigvis[viser]\"` or run `pip install \"cigvis[all]\"` to install the dependencies",
    ),
    "sliceviewer": (
        "cigvis.sliceviewer",
        "run `pip install \"cigvis[sliceviewer]\"` or `pip install panel plotly anywidget` to enable sliceviewer",
    ),
}

_LAZY_ATTRS = {
    **{name: ("vispyplot", name) for name in _VISPYPLOT_EXPORTS},
    **{name: ("mpl2dplot", name) for name in _MPL2D_EXPORTS},
    **{name: ("mpl1dplot", name) for name in _MPL1D_EXPORTS},
    "load_theme": ("mplstyle", "load_theme"),
}

_OPTIONAL_MODULE_SPECS = {
    "vispyplot": "vispy",
}


from .config import *


def _missing_optional_module(module_key):
    spec_name = _OPTIONAL_MODULE_SPECS.get(module_key)
    if spec_name and importlib.util.find_spec(spec_name) is None:
        return ImportError(f"{spec_name} not found")
    return None


def _load_lazy_module(module_key):
    module_name, message = _LAZY_MODULES[module_key]
    missing = _missing_optional_module(module_key)
    if missing is not None:
        module = ExceptionWrapper(missing, message)
        globals()[module_key] = module
        return module

    try:
        module = importlib.import_module(module_name)
    except BaseException as exc:
        if message is None:
            raise
        module = ExceptionWrapper(exc, message)

    globals()[module_key] = module
    return module


def __getattr__(name):
    if name in _LAZY_MODULES:
        return _load_lazy_module(name)

    if name in _LAZY_ATTRS:
        module_key, attr_name = _LAZY_ATTRS[name]
        module = _load_lazy_module(module_key)
        value = module if isinstance(module, ExceptionWrapper) else getattr(module, attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module 'cigvis' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "ExceptionWrapper",
    "is_running_in_notebook",
    *_CONFIG_EXPORTS,
    *_VISPYPLOT_EXPORTS,
    *_MPL2D_EXPORTS,
    *_MPL1D_EXPORTS,
    "io",
    "colormap",
    "meshs",
    "colors",
    "mplstyle",
    "mpl2dplot",
    "mpl1dplot",
    "gui",
    "vispyplot",
    "plotlyplot",
    "viserplot",
    "sliceviewer",
    "load_theme",
]
