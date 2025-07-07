# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
well logs
==============================================================

.. image:: ../../_static/cigvis/viser/03.jpg
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/viser/03.jpg'

import numpy as np
from cigvis import colormap, viserplot

ni, nx, nt = 400, 600, 360
shape = (ni, nx, nt)
data = np.fromfile('data/seis_h360x600x400.dat',
                   dtype=np.float32).reshape(shape)
nodes = viserplot.create_slices(data, pos=[20, 30, 320], cmap='gray')

points = []
for i in range(6):
    x = np.array([np.random.randint(0, ni)] * nt*4).astype(np.float32)
    y = np.array([np.random.randint(0, nx)] * nt*4).astype(np.float32)
    z = np.arange(nt*4).astype(np.float32) / 4
    if i % 3 != 0:
        v = np.random.rand(nt*4)
        points.append(np.c_[x, y, z, v])
    else:
        points.append(np.c_[x, y, z])
nodes += viserplot.create_well_logs(points, logs_type='line', width=3)


points = []
for i in range(6):
    x = np.array([np.random.randint(0, ni)] * nt*4).astype(np.float32)
    y = np.array([np.random.randint(0, nx)] * nt*4).astype(np.float32)
    z = np.arange(nt*4).astype(np.float32) / 4
    if i % 3 != 0:
        v = np.random.rand(nt*4)
        points.append(np.c_[x, y, z, v])
    else:
        points.append(np.c_[x, y, z])
nodes += viserplot.create_well_logs(points, logs_type='point', width=1)


viserplot.plot3D(nodes)
