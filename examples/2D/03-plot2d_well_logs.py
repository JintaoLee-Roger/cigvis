# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Show slice, rgt and logging curves for 2d
============================================

.. image:: ../../_static/cigvis/2D/03.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/2D/03.png'

import cigvis
import numpy as np

root = '../../'
sxp = 'data/rgt/sx.dat'
lxp = 'data/rgt/ux.dat'
fxp = 'data/rgt/fx.dat'
ni, nx, nt = 128, 128, 128

sx = np.memmap(root + sxp, np.float32, 'r', shape=(ni, nx, nt))
lx = np.memmap(root + lxp, np.float32, 'r', shape=(ni, nx, nt))
sx2 = sx[20, :, :]
lx2 = lx[20, :, :]

las = cigvis.io.load_las('../../data/cb23.las')['data'][:2000, 3]
x = 80  # assume x location is 80

fg = {}
fg['img'] = cigvis.fg_image_args(lx2, alpha=0.5)

w = 20
y = np.linspace(0, 125, len(las))
# normalize well log curve
# las -> [-0.5, 0.5]
las = (las - las.min()) / (las.max() - las.min()) - 0.5
# las -> [-0.5*w+x, 0.5*w+x]
x = las * w + x
fg['line'] = cigvis.line_args(x, y, 'black')

cigvis.plot2d(sx2,
              fg,
              figsize=(6, 6),
              xlabel='xline',
              ylabel='time',
              title='Example',
              save='example.png',
              dpi=200)
