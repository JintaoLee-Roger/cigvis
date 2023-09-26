# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
设置坐标轴范围
==================================

将y轴设置为时间轴, 起始为 1s, 间隔 2ms,
将x轴设置为x坐标, 起始为 4098, 间隔 12.5m

.. image:: ../../_static/cigvis/2D/10.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/2D/10.png'

import cigvis
import numpy as np

d = np.fromfile('../../data/rgt/sx.dat', np.float32).reshape(128, 128, 128)
sl = d[30, :, :]

cigvis.plot2d(
    sl,
    figsize=(6, 6),
    xlabel='xline/m',
    ylabel='time/s',
    title='set tick labels',
    xsample=[4098, 12.5],  # [start, step]
    ysample=[1, 0.02],  # [start, step]
    cbar='Amplitude',
    save='example.png',
    dpi=200)
