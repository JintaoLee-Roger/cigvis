# Copyright (c) 2026 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Bodies and points in Viser
==========================

Display a browser-based 3D scene with volume slices, an isosurface body, and
sparse pick/marker points.
"""

import numpy as np
from pathlib import Path

from cigvis import viserplot


root = Path(__file__).resolve().parent.parent.parent

seisp = root / 'data/co2/sx.dat'
ni, nx, nt = 192, 192, 240
sx = np.fromfile(seisp, np.float32).reshape(ni, nx, nt)

attribute = np.abs(sx)
level = np.percentile(attribute, 99.2)

points = np.array([
    [48, 46, 72, 0.1],
    [82, 120, 110, 0.5],
    [132, 80, 150, 0.9],
    [160, 155, 190, 1.0],
], dtype=np.float32)

nodes = viserplot.create_slices(sx, pos=[40, 80, 180], cmap='gray')
nodes += viserplot.create_bodies(
    attribute,
    level=level,
    margin=0.0,
    color='cyan',
    filter_sigma=1,
)
nodes += viserplot.create_points(
    points,
    color=None,
    cmap='jet',
    clim=[0, 1],
    r=6,
    point_shape='circle',
)

viserplot.plot3D(nodes)
