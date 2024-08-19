# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Display 3D seismic data and multiple horizons
================================================

层位可以是一个 (n1, n2) 大小的 Z 值,
也可以是 (N, 3) 大小的 N 个点

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/3Dvispy/05.png'

import numpy as np
from cigvis import viserplot

sxp = 'data/co2/sx.dat'
sfp1 = 'data/co2/mh21.dat'
sfp2 = 'data/co2/mh22.dat'
ni, nx, nt = 192, 192, 240

sx = np.fromfile(sxp, np.float32).reshape(ni, nx, nt)
sf1 = np.fromfile(sfp1, np.float32).reshape(ni, nx)
sf2 = np.fromfile(sfp2, np.float32).reshape(ni, nx)

nodes = viserplot.create_slices(sx, cmap='gray')

# show amplitude
nodes += viserplot.create_surfaces(
    [sf1, sf2],
    volume=sx,
    value_type='amp',
    cmap='gray',
    clim=[sx.min(), sx.max()],
)

# # add two points
# nodes += viserplot.create_points(np.array([[70, 50, 158], [20, 100, 80]]), r=3)

viserplot.plot3D(nodes)
