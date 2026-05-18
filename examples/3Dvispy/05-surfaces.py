# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Display 3D seismic data and multiple horizons
================================================

Horizons can be either an (n1, n2) array of Z values or an (N, 3) point cloud.

.. image:: ../../_static/cigvis/3Dvispy/05.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/3Dvispy/05.png'

import numpy as np
import cigvis
from pathlib import Path
root = Path(__file__).resolve().parent.parent.parent

sxp = root / 'data/co2/sx.dat'
sfp1 = root / 'data/co2/mh21.dat'
sfp2 = root / 'data/co2/mh22.dat'
ni, nx, nt = 192, 192, 240

sx = np.fromfile(sxp, np.float32).reshape(ni, nx, nt)
sf1 = np.fromfile(sfp1, np.float32).reshape(ni, nx)
sf2 = np.fromfile(sfp2, np.float32).reshape(ni, nx)

nodes = cigvis.create_slices(sx)

# show amplitude
nodes += cigvis.create_surfaces([sf1, sf2],
                                volume=sx,
                                value_type='amp',
                                cmap='Petrel',
                                clim=[sx.min(), sx.max()])

# add two points
nodes += cigvis.create_points(np.array([[70, 50, 158], [20, 100, 80]]), r=3)

cigvis.plot3D(
    nodes,
    view=cigvis.Plot3DView(size=(800, 800)),
    save=cigvis.Plot3DSave(path='example.png'),
)
