# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Show 2d contours of slice, rgt fault and rgt
================================================

Use ``skimage.measure.find_countours`` to calculate contours

.. image:: ../../_static/cigvis/2D/07.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/2D/07.png'

import cigvis
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours

root = '../../'
sxp = 'data/rgt/sx.dat'
lxp = 'data/rgt/ux.dat'
ni, nx, nt = 128, 128, 128

sx = np.memmap(root + sxp, np.float32, 'r', shape=(ni, nx, nt))
lx = np.memmap(root + lxp, np.float32, 'r', shape=(ni, nx, nt))
sx2 = sx[20, :, :]
lx2 = lx[20, :, :]

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

fg = {}

# rgt overlaid
fg['img'] = cigvis.fg_image_args(lx2, alpha=0.5)

# find rgt's isolines at lx2[20, 20], lx2[60, 60], and lx2[100, 100]
isos1 = find_contours(lx2, lx2[20, 20])[0]
isos2 = find_contours(lx2, lx2[60, 60])[0]
isos3 = find_contours(lx2, lx2[100, 100])[0]
fg['line'] = cigvis.line_args(isos3[:, 0], isos3[:, 1], lw=3)
fg['line'] += cigvis.line_args(isos2[:, 0], isos2[:, 1], lw=3)
fg['line'] += cigvis.line_args(isos1[:, 0], isos1[:, 1], lw=3)

# add markers at [20, 20], [60, 60], and [100, 100]
fg['marker'] = cigvis.marker_args([20, 60, 100], [20, 60, 100], c='red', s=40)

cigvis.plot2d(sx2,
              fg,
              xlabel='xline',
              ylabel='time',
              title="don't show legend",
              ax=axes[0])

fg['line'] = cigvis.line_args(isos3[:, 0],
                              isos3[:, 1],
                              lw=3,
                              label='pos=(20,20)')
fg['line'] += cigvis.line_args(isos2[:, 0],
                               isos2[:, 1],
                               lw=3,
                               label='pos=(60,60)')
fg['line'] += cigvis.line_args(isos1[:, 0],
                               isos1[:, 1],
                               lw=3,
                               label='pos=(60,60)')
cigvis.plot2d(sx2,
              fg,
              xlabel='xline',
              ylabel='time',
              title='show legend (must set `label=`)',
              show_legend=True,
              ax=axes[1])

plt.tight_layout()
plt.savefig('example.png', bbox_inches='tight', pad_inches=0.01, dpi=200)
plt.show()
