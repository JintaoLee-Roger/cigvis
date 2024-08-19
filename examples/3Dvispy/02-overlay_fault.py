# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Overlaying fault displays on slices of 3D seismic data bodies
===================================================================

``create_overlay``: the first parameters is (background), 
and the second parameters is (foreground)

.. Note::
    foreground 需要合理设置透明度和mask

.. image:: ../../_static/cigvis/3Dvispy/02.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/3Dvispy/02.png'

import numpy as np
import cigvis
from cigvis import colormap
from pathlib import Path
root = Path(__file__).resolve().parent.parent.parent

sxp = root / 'data/rgt/sx.dat'
fxp = root / 'data/rgt/fx.dat'
ni, nx, nt = 128, 128, 128

sx = np.fromfile(sxp, np.float32).reshape(ni, nx, nt)
fx = np.fromfile(fxp, np.float32).reshape(ni, nx, nt)

# mask min value (0), 0 means no fault
fg_cmap = colormap.set_alpha_except_min('jet', alpha=1)

# fx is discrete data, set interpolation as 'nearest'
nodes = cigvis.create_slices(sx, pos=[[36], [28], [84]], cmap='gray')
nodes = cigvis.add_mask(nodes, fx, cmaps=fg_cmap, interpolation='nearest')
nodes += cigvis.create_colorbar_from_nodes(nodes, 'Amplitude', select='slices')

cigvis.plot3D(nodes, size=(800, 600), savename='example.png')
