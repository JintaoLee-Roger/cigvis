# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
plot a 1D trace
================

This example demonstrates how to plot a 1D trace

.. image:: ../../_static/cigvis/1D/0.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/1D/0.png'

import numpy as np
import cigvis

root = '/Users/lijintao/work/mygit/pyseisview/data/'
sxp = root + 'seis_h360x600x400.dat'
ni, nx, nt = 400, 600, 360

sx = np.fromfile(sxp, np.float32).reshape(ni, nx, nt)

trace = sx[100, 100, :]

# 垂直显示
cigvis.plot1d(trace,
              dt=0.02,
              axis_label='Time / s',
              c='skyblue',
              save='example.png',
              dpi=200)
