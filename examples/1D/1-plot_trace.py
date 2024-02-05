# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
fill 模式
=======================

This example demonstrates how to plot a 1D trace

.. image:: ../../_static/cigvis/1D/1.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/1D/1.png'

import numpy as np
import cigvis
import matplotlib.pyplot as plt

root = '../../data/'
sxp = root + 'seis_h360x600x400.dat'
ni, nx, nt = 400, 600, 360

sx = np.fromfile(sxp, np.float32).reshape(ni, nx, nt)

trace = sx[100, 100, :]

# 垂直显示
# cigvis.plot1d(trace, dt=0.02, axis_label='Time / s', c='skyblue')

fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# 水平显示
cigvis.plot1d(trace, orient='h', dt=0.02, ax=axes[0])

# 填充模式

# 填充波峰
cigvis.plot1d(trace,
              orient='h',
              dt=0.02,
              fill_up=0.3,
              value_label='Amplitude',
              ax=axes[1])

# 填充波谷
cigvis.plot1d(trace,
              orient='h',
              dt=0.02,
              axis_label='Time / s',
              fill_down=0.3,
              ax=axes[2])

plt.tight_layout()
plt.savefig('example.png', bbox_inches='tight', pad_inches=0.01, dpi=200)
plt.show()
