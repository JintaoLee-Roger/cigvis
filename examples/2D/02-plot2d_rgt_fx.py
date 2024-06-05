# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
show slice, rgt, fault
==========================

.. image:: ../../_static/cigvis/2D/02.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/2D/02.png'

import cigvis
import numpy as np
from cigvis import colormap

root = '../../'
sxp = 'data/rgt/sx.dat'
lxp = 'data/rgt/ux.dat'
fxp = 'data/rgt/fx.dat'
ni, nx, nt = 128, 128, 128

sx = np.memmap(root + sxp, np.float32, 'r', shape=(ni, nx, nt))
lx = np.memmap(root + lxp, np.float32, 'r', shape=(ni, nx, nt))
fx = np.memmap(root + fxp, np.float32, 'r', shape=(ni, nx, nt))
sx2 = sx[20, :, :]
lx2 = lx[20, :, :]
fx2 = fx[20, :, :]

fg = {}
# rgt
fg['img'] = cigvis.fg_image_args(lx2, alpha=0.5, show_cbar=False)

# fault
fx_cmap = colormap.set_alpha_except_min('jet', 1, False)
fg['img'] += cigvis.fg_image_args(fx2,
                                  fx_cmap,
                                  interpolation='nearest',
                                  show_cbar=True)

cigvis.plot2d(sx2,
              fg,
              figsize=(6, 6),
              xlabel='xline',
              ylabel='time',
              title='Example',
              cbar='fault',
              discrete=True,
              save='example.png',
              dpi=200)
