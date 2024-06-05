# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Display discrete colorbar
========================================

.. image:: ../../_static/cigvis/2D/08.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/2D/08.png'

import numpy as np
import cigvis

root = '../../data/'
sxp = root + 'seis_h360x600x400.dat'
lxp = root + 'label_h360x600x400.dat'
ni, nx, nt = 400, 600, 360

sx = np.memmap(sxp, np.float32, 'r', shape=(ni, nx, nt))
lx = np.memmap(lxp, np.float32, 'r', shape=(ni, nx, nt))

sx2 = sx[:, 100, :]
lx2 = lx[:, 100, :]

fg = {}

fg['img'] = cigvis.fg_image_args(lx2, alpha=0.5, interpolation='nearest')

cigvis.plot2d(
    sx2,
    fg,
    cbar='Facies',
    discrete=True,
    tick_labels=['Class A', 'Class B', 'Class C', 'Class D', 'Class E'],
    save='example.png',
    dpi=200)
