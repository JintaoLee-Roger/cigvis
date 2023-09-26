# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
划分多个canvas, 显示多个3D数据
========================================

.. image:: ../../_static/cigvis/3Dvispy/10.gif
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/3Dvispy/10.png'

import numpy as np
import cigvis

seisp = '../../data/co2/sx.dat'
ni, nx, nt = 192, 192, 240
sx = np.fromfile(seisp, np.float32).reshape(ni, nx, nt)

nodes1 = cigvis.create_slices(sx, cmap='Petrel')
nodes2 = cigvis.create_slices(sx, cmap='Petrel')

cigvis.plot3D(
    [nodes1, nodes2],
    grid=(1, 2),  # here, define a grid
    size=(1000, 800),
    savename='example.png')
