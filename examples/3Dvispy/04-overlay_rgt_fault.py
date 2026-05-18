# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Overlay multiple 3D data bodies (RGT, fault) on slices of 3D seismic data bodies
====================================================================================

Use ``create_slices`` for the background and call ``add_mask`` once per
foreground overlay volume.

.. Note::
    Set foreground transparency and masking carefully.

.. image:: ../../_static/cigvis/3Dvispy/04.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/3Dvispy/04.png'

import numpy as np
import cigvis
from cigvis import colormap
from pathlib import Path
root = Path(__file__).resolve().parent.parent.parent

sxp = root / 'data/rgt/sx.dat'
uxp = root / 'data/rgt/ux.dat'
fxp = root / 'data/rgt/fx.dat'
ni, nx, nt = 128, 128, 128

sx = np.fromfile(sxp, np.float32).reshape(ni, nx, nt)
rgt = np.fromfile(uxp, np.float32).reshape(ni, nx, nt)
fx = np.fromfile(fxp, np.float32).reshape(ni, nx, nt)

nodes = cigvis.create_slices(sx, pos=[[36], [28], [84]], cmap='gray')

nodes = cigvis.add_mask(nodes, rgt, cmap='jet', alpha=0.4)

# fx is discrete data, set interpolation as 'nearest'
nodes = cigvis.add_mask(nodes, fx, cmap='jet', interpolation='nearest', alpha=1, excpt='min')

nodes += cigvis.create_colorbar_from_nodes(nodes, 'RGT', select='mask', idx=0) # idx = 0 means the first mask

cigvis.plot3D(
    nodes,
    view=cigvis.Plot3DView(size=(750, 600)),
    save=cigvis.Plot3DSave(path='example.png'),
    gui=True
)
