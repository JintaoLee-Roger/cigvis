# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Share camera parameters across multiple canvas
================================================

This makes it easy to compare two data results.

.. image:: ../../_static/cigvis/3Dvispy/11.gif
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/3Dvispy/11.png'

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
    view=cigvis.Plot3DView(
        grid=(1, 2),  # here, define a grid
        share=True,  # here, link all cameras
        size=(1000, 800),
    ),
    save=cigvis.Plot3DSave(path='example.png'),
)
