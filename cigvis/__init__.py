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
            self.exception = type(e)(f"{e.args[0]}\n\t{custom}", *e.args[1:])
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
    except NameError:
        return False


import sys
from .config import *
from . import io
from . import colormap
from . import meshs
from . import gui

injupyter = is_running_in_notebook()

if injupyter:
    from .plotlyplot import *
else:
    from .vispyplot import *

try:
    from . import viserplot
except BaseException as E:
    viserplot = ExceptionWrapper(
        E,
        "run `pip install \"cigvis[viser]\"` or run `pip install \"cigvis[all]\"` to install the dependencies"
    )

from .mpl2dplot import *
from .mpl1dplot import *
from . import colors
from .mplstyle import load_theme
