# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
The log track is displayed as a tube and multiple log curves are displayed
============================================================================

This example demonstrates how to visualize a well
log with multi traces.

The first trace is plotted as a tube, the others 
are plotted as faces.

.. image:: ../../_static/cigvis/3Dvispy/08.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/3Dvispy/08.png'

import numpy as np
import cigvis
from pathlib import Path
root = Path(__file__).resolve().parent.parent.parent

sxp = root / 'data/co2/sx.dat'
lasp = root / 'data/cb23.las'
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

# idx=0 means select the first WellLog node in nodes (though there is only one WellLog node in this case).
# idx2=1 means select the second cmap, and clim for the select WellLog node, ('seismic' in this case).
# nodes += cigvis.create_colorbar_from_nodes(nodes, 'Log Impedance', select='logs', idx=0, idx2=1)
nodes += cigvis.create_axis(sx.shape, 'axis', 'auto', axis_labels=['Inline [km]', 'Xline [km]', 'Time [s]'], line_width=1, intervals=[0.025, 0.025, 0.002], rotation=(30, -30, -90), tick_nums=4)

cigvis.plot3D(nodes, zoom_factor=8, size=(800, 600), savename='example.png', xyz_axis=False)
