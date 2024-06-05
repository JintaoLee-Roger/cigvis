# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Displaying multiple images using subplots
============================================

.. image:: ../../_static/cigvis/2D/05.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/2D/05.png'

import cigvis
import numpy as np
import matplotlib.pyplot as plt

sx = np.fromfile('../../data/rgt/sx.dat', np.float32).reshape(128, 128, 128)
lx = np.fromfile('../../data/rgt/ux.dat', np.float32).reshape(128, 128, 128)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))

fg = dict(img=cigvis.fg_image_args(lx[30, :, :], alpha=0.5))
cigvis.plot2d(sx[30, :, :], fg, title='inline=30', ax=axes[0, 0])

fg = dict(img=cigvis.fg_image_args(lx[60, :, :], alpha=0.5))
cigvis.plot2d(sx[60, :, :], fg, title='inline=60', ax=axes[0, 1])

fg = dict(img=cigvis.fg_image_args(lx[90, :, :], alpha=0.5))
cigvis.plot2d(sx[90, :, :], fg, title='inline=90', ax=axes[1, 0])

fg = dict(img=cigvis.fg_image_args(lx[120, :, :], alpha=0.5))
cigvis.plot2d(sx[120, :, :], fg, title='inline=120', ax=axes[1, 1])

plt.tight_layout()
plt.savefig('example.png', bbox_inches='tight', pad_inches=0.01, dpi=200)

plt.show()