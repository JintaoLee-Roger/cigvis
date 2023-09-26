# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
显示离散的 colorbar, 并设置 mask
========================================

.. image:: ../../_static/cigvis/2D/09.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/2D/09.png'

import numpy as np
import cigvis
from cigvis import colormap
import matplotlib.pyplot as plt

root = '/Users/lijintao/work/mygit/pyseisview/data/'
sxp = root + 'seis_h360x600x400.dat'
lxp = root + 'label_h360x600x400.dat'
ni, nx, nt = 400, 600, 360

sx = np.memmap(sxp, np.float32, 'r', shape=(ni, nx, nt))
lx = np.memmap(lxp, np.float32, 'r', shape=(ni, nx, nt))

sx2 = sx[:, 100, :]
lx2 = lx[:, 100, :]
ticks_label = ['Class A', 'Class B', 'Class C', 'Class D', 'Class E']

fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))

fg = {}
fg['img'] = cigvis.fg_image_args(lx2, alpha=0.5, interpolation='nearest')
cigvis.plot2d(sx2,
              fg,
              cbar='Facies',
              discrete=True,
              tick_labels=ticks_label,
              title='discrete colorbar',
              ax=axes[0, 0])

cmap = colormap.set_alpha_except_min('jet', 0.5, False)
fg['img'] = cigvis.fg_image_args(lx2, cmap=cmap, interpolation='nearest')
cigvis.plot2d(sx2,
              fg,
              cbar='Facies',
              discrete=True,
              tick_labels=ticks_label,
              title='remove the min value',
              ax=axes[0, 1])

cmap = colormap.set_alpha_except_min('jet', 0.5, False)
fg['img'] = cigvis.fg_image_args(lx2, cmap=cmap, interpolation='nearest')
cigvis.plot2d(sx2,
              fg,
              cbar='Facies',
              discrete=True,
              tick_labels=ticks_label,
              remove_trans=False,
              title='cbar keep the min value',
              ax=axes[1, 0])

colors = ['red', 'green', 'yellow', 'blue', (0, 0.5, 0.5)]
values = np.unique(lx2)
cmap = colormap.custom_disc_cmap(values, colors)
cmap = colormap.set_alpha_except_min(cmap, 0.5, False)
fg['img'] = cigvis.fg_image_args(lx2, cmap=cmap, interpolation='nearest')
cigvis.plot2d(sx2,
              fg,
              cbar='Facies',
              discrete=True,
              tick_labels=ticks_label,
              title='custom cmap',
              ax=axes[1, 1])

plt.tight_layout()
plt.savefig('example.png', bbox_inches='tight', pad_inches=0.01, dpi=200)

plt.show()
