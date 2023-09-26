# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
显示 2d 的 slice, rgt
================================

可以合理设置cmap来mask一些区域

.. image:: ../../_static/cigvis/2D/06.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/2D/06.png'

import cigvis
import numpy as np
import matplotlib.pyplot as plt
from cigvis import colormap

sx = np.fromfile('../../data/rgt/sx.dat', np.float32).reshape(128, 128, 128)
lx = np.fromfile('../../data/rgt/ux.dat', np.float32).reshape(128, 128, 128)

sx2 = sx[30, :, :]
lx2 = lx[30, :, :]

fg1 = {}
fg1['img'] = cigvis.fg_image_args(lx2, alpha=0.5)

fg2 = {}
fg2_cmap = colormap.set_alpha_except_max('jet', 0.5, False)
fg2['img'] = cigvis.fg_image_args(lx2,
                                  fg2_cmap,
                                  clim=[lx2.min(), lx2.max() * 0.8])

fg3 = {}
fg3_cmap = colormap.set_alpha_except_top('jet',
                                         0.5,
                                         clim=[lx2.min(), lx2.max()],
                                         segm=lx2.max() * 0.8,
                                         forvispy=False)
fg3['img'] = cigvis.fg_image_args(lx2, fg3_cmap)

fg4 = {}
fg4_cmap = colormap.set_alpha_except_ranges('jet',
                                            0.5,
                                            [lx2.min(), lx2.max()],
                                            r=[[120, 130], [140, 150]],
                                            forvispy=False)
fg4['img'] = cigvis.fg_image_args(lx2, fg4_cmap)

fig, axs = plt.subplots(2, 2, figsize=(8.5, 7))

cigvis.plot2d(sx2,
              fg1,
              ax=axs[0, 0],
              cbar='rgt',
              xlabel='xline',
              ylabel='time',
              title='overlid')
cigvis.plot2d(sx2,
              fg2,
              ax=axs[0, 1],
              cbar='rgt',
              xlabel='xline',
              ylabel='time',
              title='except top 1')
cigvis.plot2d(sx2,
              fg3,
              ax=axs[1, 0],
              cbar='rgt',
              xlabel='xline',
              ylabel='time',
              title='except top 2')
cigvis.plot2d(sx2,
              fg4,
              ax=axs[1, 1],
              cbar='rgt',
              xlabel='xline',
              ylabel='time',
              title='except range')
plt.tight_layout()
plt.savefig('example.png', bbox_inches='tight', pad_inches=0.01, dpi=200)

plt.show()
