# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Overlay RGT display on slice of 3D seismic data volume
==========================================================

Use ``create_slices`` for the background and ``add_mask`` for the RGT
overlay.

.. Note::
    Set foreground transparency and masking carefully.

.. image:: ../../_static/cigvis/3Dvispy/03.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/3Dvispy/03.png'

import numpy as np
import cigvis
from cigvis import colormap
from pathlib import Path
root = Path(__file__).resolve().parent.parent.parent

sxp = root / 'data/rgt/sx.dat'
uxp = root / 'data/rgt/ux.dat'
ni, nx, nt = 128, 128, 128

sx = np.fromfile(sxp, np.float32).reshape(ni, nx, nt)
rgt = np.fromfile(uxp, np.float32).reshape(ni, nx, nt)

nodes = cigvis.create_slices(sx, pos=[[36], [28], [84]], cmap='gray')

nodes = cigvis.add_mask(nodes, rgt, cmap='jet', alpha=0.4)
# this is equivalent to
# fg_cmap = colormap.set_alpha('jet', alpha=0.4)
# nodes = cigvis.add_mask(nodes, rgt, cmaps=fg_cmap)


nodes += cigvis.create_colorbar_from_nodes(nodes, 'RGT', select='mask')

cigvis.plot3D(
    nodes,
    view=cigvis.Plot3DView(size=(750, 600)),
    save=cigvis.Plot3DSave(path='example.png', transparent_bg=False),
)
