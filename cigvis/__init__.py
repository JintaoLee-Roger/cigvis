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

import sys
from .config import *
from . import io
from . import colormap
from . import meshs
from . import gui

injupyter = 'ipykernel_launcher.py' in sys.argv[0] or 'lab' in sys.argv[0]

if injupyter:
    from .plotlyplot import *
else:
    from .vispyplot import *

from .mpl2dplot import *
from .mpl1dplot import *