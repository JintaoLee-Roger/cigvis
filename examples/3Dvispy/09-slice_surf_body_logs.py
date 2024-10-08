# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Displays a variety of geophysical data
================================================

.. image:: ../../_static/cigvis/3Dvispy/09.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/3Dvispy/09.png'

import numpy as np
import cigvis
from pathlib import Path
root = Path(__file__).resolve().parent.parent.parent

sx = np.memmap(root / 'data/co2/sx.dat', np.float32, 'r', shape=(192, 192, 240))
lx = np.memmap(root / 'data/co2/lx.dat', np.float32, 'c', shape=(192, 192, 240))
sf1 = np.fromfile(root / 'data/co2/mh21.dat', np.float32).reshape(192, 192)
sf2 = np.fromfile(root / 'data/co2/mh22.dat', np.float32).reshape(192, 192)

# v = np.random.rand(100)
las = cigvis.io.load_las(root / 'data/cb23.las')
idx = las['Well']['name'].index('NULL')
null_value = float(las['Well']['value'][3])
v = las['data'][:, 1:5]

x = np.linspace(50, 100, len(v))
y = np.linspace(50, 100, len(v))
z = np.sin((x - 50) / 100 * np.pi) * 200

points = np.c_[x, y, z]

nodes = []
nodes += cigvis.create_slices(sx)
nodes += cigvis.create_well_logs(
    points,
    v,
    cyclinder=False,
    null_value=null_value,
    cmap=['jet', 'seismic', 'Petrel', 'od_seismic1'])
nodes += cigvis.create_points(np.array([[70, 50, 158], [20, 100, 80]]), r=3)
nodes += cigvis.create_bodys(lx, 0.5, 0)
nodes += cigvis.create_surfaces([sf1, sf2],
                                sx,
                                'amp',
                                cmap='Petrel',
                                clim=[sx.min(), sx.max()])
nodes += cigvis.create_colorbar_from_nodes(nodes, 'Amplitude', select='slices')
cigvis.plot3D(nodes, savename='example.png')
