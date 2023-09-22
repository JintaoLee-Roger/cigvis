# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
将测井轨迹显示为一个管子 (tube)
=======================================

"""

import numpy as np
import cigvis

sxp = '../../data/co2/sx.dat'
ni, nx, nt = 192, 192, 240

sx = np.fromfile(sxp, np.float32).reshape(ni, nx, nt)

x = np.linspace(50, 100, 5000)
y = np.linspace(50, 150, 5000)
z = np.sin((y - 50) / 200 * np.pi) * 200

points = np.c_[x, y, z]

nodes = cigvis.create_slices(sx)
nodes += cigvis.create_well_logs(points, cmap='red')

cigvis.plot3D(nodes, size=(800, 800), savename='example.png')