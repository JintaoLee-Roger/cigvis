# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Display a 3D volume of data (by selecting several slices)
============================================================

.. image:: ../../_static/cigvis/3Dvispy/01.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/3Dvispy/01.png'

import numpy as np
import cigvis

seisp = '../../data/co2/sx.dat'
ni, nx, nt = 192, 192, 240
sx = np.fromfile(seisp, np.float32).reshape(ni, nx, nt)

nodes, cbar = cigvis.create_slices(sx,
                                   cmap='Petrel',
                                   return_cbar=True,
                                   label_str='Amplitude')
nodes.append(cbar)

cigvis.plot3D(nodes, size=(1000, 800), savename='example.png')
