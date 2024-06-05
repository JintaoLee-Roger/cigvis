# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Set font
================

.. image:: ../../_static/cigvis/2D/11.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/2D/11.png'

import matplotlib
import cigvis
import numpy as np
import matplotlib.pyplot as plt

# set font family
font = {'family': 'Times New Roman', 'size': 12}
matplotlib.rc('font', **font)

d = np.fromfile('../../data/rgt/sx.dat', np.float32).reshape(128, 128, 128)
sl = d[30, :, :]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

cigvis.plot2d(sl,
              xlabel='xline',
              ylabel='time',
              title='origin',
              cbar='Amplitude',
              ax=axes[0])

cigvis.plot2d(sl,
              xlabel='xline',
              ylabel='time',
              title='set font size',
              cbar='Amplitude',
              title_size=16,
              xlabel_size=14,
              ylabel_size=14,
              ticklabels_size=12,
              cbar_label_size=14,
              cbar_ticklabels_size=12,
              ax=axes[1])

plt.tight_layout()

plt.savefig('example.png', bbox_inches='tight', pad_inches=0.01, dpi=200)

plt.show()
