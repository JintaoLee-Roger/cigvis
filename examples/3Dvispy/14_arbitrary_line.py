# Copyright (c) 2024 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Display an arbitray line
============================================================

.. image:: ../../_static/cigvis/3Dvispy/14.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/3Dvispy/14.png'

import numpy as np
import cigvis
from pathlib import Path
root = Path(__file__).resolve().parent.parent.parent

seisp = root / 'data/co2/sx.dat'
ni, nx, nt = 192, 192, 240
sx = np.fromfile(seisp, np.float32).reshape(ni, nx, nt)

nodes = cigvis.create_slices(sx, cmap='Petrel')
nodes += cigvis.create_arbitrary_line(anchor=[[0, 0], [90, 190], [190, 50]],
                                      volume=sx,
                                      nodes=nodes)

cigvis.plot3D(nodes, size=(1000, 800), savename='example.png')
