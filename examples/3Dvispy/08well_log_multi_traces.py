# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
将测井轨迹显示为一个tube, 并显示多条测井曲线
===============================================

This example demonstrates how to visualize a well
log with multi traces.

The first trace is plotted as a tube, the others 
are plotted as faces.
"""

import numpy as np
import cigvis

sxp = '../../data/co2/sx.dat'
lasp = '../../data/cb23.las'
ni, nx, nt = 192, 192, 240

sx = np.fromfile(sxp, np.float32).reshape(ni, nx, nt)
las = cigvis.io.load_las(lasp)
idx = las['Well']['name'].index('NULL')
null_value = float(las['Well']['value'][3])
v = las['data'][:, 1:5]

x = np.linspace(50, 100, len(v))
y = np.linspace(50, 150, len(v))
z = np.sin((y - 50) / 200 * np.pi) * 200

points = np.c_[x, y, z]

nodes = cigvis.create_slices(sx)

# cyclinder=False: 半径和颜色都表示测井值的大小
# cyclinder=True: 井柱半径不变
nodes += cigvis.create_well_logs(
    points,
    v,
    cyclinder=False,
    null_value=null_value,
    cmap=['jet', 'seismic', 'Petrel', 'od_seismic1'])

cigvis.plot3D(nodes, zoom_factor=8, size=(800, 800), savename='example.png')
