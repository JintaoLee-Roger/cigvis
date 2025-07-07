# Copyright (c) 2025 Jintao Li. 
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
viser comparision
====================

A demo to comarison different results in a browser.

.. image:: ../../_static/cigvis/viser/04.gif
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/viser/04.jpg'


import numpy as np
from cigvis import viserplot
from pathlib import Path
root = Path(__file__).resolve().parent.parent.parent

sxp = root / 'data/rgt/sx.dat'
fxp = root / 'data/rgt/fx.dat'
ni, nx, nt = 128, 128, 128

sx = np.fromfile(sxp, np.float32).reshape(ni, nx, nt)
fx = np.fromfile(fxp, np.float32).reshape(ni, nx, nt)

s1 = viserplot.create_server(8080)
s2 = viserplot.create_server(8081)

nodes1 = viserplot.create_slices(sx, cmap='gray', pos=[20, 20, 100])
nodes2 = viserplot.create_slices(sx, cmap='gray', pos=[20, 20, 100])
nodes2 = viserplot.add_mask(nodes2, fx, cmaps='jet', alpha=1, excpt='min')

viserplot.plot3D(nodes1, server=s1, run_app=False)
viserplot.plot3D(nodes2, server=s2, run_app=False)
viserplot.run()