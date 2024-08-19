# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Divide multiple canvas to display multiple 3D data
======================================================

.. image:: ../../_static/cigvis/3Dvispy/10.gif
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/3Dvispy/10.png'

import numpy as np
import cigvis
from pathlib import Path
root = Path(__file__).resolve().parent.parent.parent

seisp = root / 'data/co2/sx.dat'
ni, nx, nt = 192, 192, 240
sx = np.fromfile(seisp, np.float32).reshape(ni, nx, nt)

nodes1 = cigvis.create_slices(sx, cmap='Petrel')
nodes2 = cigvis.create_slices(sx, cmap='Petrel')

cigvis.plot3D(
    [nodes1, nodes2],
    grid=(1, 2),  # here, define a grid
    size=(1000, 800),
    savename='example.png')
