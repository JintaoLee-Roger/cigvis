# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
CIGVIS
=======

``cigvis`` is a tool for geophysical data visualization, which developed by 
`Computational Interpretation Group (CIG) <https://cig.ustc.edu.cn/main.htm>`_.

``cigvis`` can achieve a variety of geophysical data visualization, including:
3D volumes (e.g., seismic) and its overlay visual (e.g., 3D label), 
surfaces, logs, points, geological bodys, 2D images, one 1D trace, multi-traces.

``cigvis`` can be used in both desktop and jupyter environment.

In desktop environment, it is based on `vispy <https://github.com/vispy/vispy>`_ 
and wild mixtures of `seismic_canvas <https://github.com/yunzhishi/seismic-canvas/tree/master>`_.

In jupyter environment, it is based on `plotly <https://github.com/plotly/plotly.py>`_.

``cigvis`` also provides some extenal colormaps in matplotlib's Colormap format.
These custom colormap is converted from OpendTect software (version is 7.0.0).
"""

import sys
from .config import *
from . import io
from . import colormap

injupyter = 'ipykernel_launcher.py' in sys.argv[0] or 'lab' in sys.argv[0]

if injupyter:
    from .plotlyplot import *
else:
    from .vispyplot import *

from .plot1d2d import *