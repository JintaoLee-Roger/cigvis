# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Display a 3D volume of data (by selecting several slices)
============================================================

.. image:: ../../_static/cigvis/3Dvispy/01.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/3Dvispy/01.png'

import numpy as np
import cigvis
from pathlib import Path
root = Path(__file__).resolve().parent.parent.parent

seisp = root / 'data/co2/sx.dat'
ni, nx, nt = 192, 192, 240
sx = np.fromfile(seisp, np.float32).reshape(ni, nx, nt)

nodes = cigvis.create_slices(sx, cmap='Petrel')
nodes += cigvis.create_colorbar_from_nodes(nodes, 'Amplitude', select='slices')

cigvis.plot3D(nodes, size=(700, 600), savename='example.png', xyz_axis=False)
