# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Display 3D seismic data and 3D geological bodies (CO2)
=======================================================

.. image:: ../../_static/cigvis/3Dvispy/06.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/3Dvispy/06.png'

import numpy as np
import cigvis
from pathlib import Path
root = Path(__file__).resolve().parent.parent.parent

sxp = root / 'data/co2/sx.dat'
lxp = root / 'data/co2/lx.dat'
ni, nx, nt = 192, 192, 240

sx = np.fromfile(sxp, np.float32).reshape(ni, nx, nt)
lx = np.fromfile(lxp, np.float32).reshape(ni, nx, nt)


nodes = cigvis.create_slices(sx)

nodes += cigvis.create_bodys(lx, level=0.5, margin=0, filter_sigma=1)

cigvis.plot3D(nodes, size=(800, 800), savename='example.png')
