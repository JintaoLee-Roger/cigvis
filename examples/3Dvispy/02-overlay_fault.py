# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Overlaying fault displays on slices of 3D seismic data bodies
===================================================================

Use ``create_slices`` for the background and ``add_mask`` for the fault
overlay.

.. Note::
    Set foreground transparency and masking carefully.

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

coords = np.argwhere(fx > 0).astype(np.float32)


# fx is discrete data, set interpolation as 'nearest'
nodes = cigvis.create_slices(sx, pos=[[36], [28], [84]], cmap='gray')

# mask min value (0), 0 means no fault, 
# excpt='min' means that set minimum value to transparent
nodes = cigvis.add_mask(nodes, fx, cmap='jet', interpolation='nearest', alpha=1, excpt='min')
# this is equivalent to
# fg_cmap = colormap.set_alpha_except_min('jet', alpha=1)
# nodes = cigvis.add_mask(nodes, fx, cmaps=fg_cmap, interpolation='nearest')

nodes += cigvis.create_colorbar_from_nodes(nodes, 'Amplitude', select='slices')

cigvis.plot3D(
    nodes,
    view=cigvis.Plot3DView(size=(700, 600)),
    save=cigvis.Plot3DSave(path='example.png', transparent_bg=False),
)
