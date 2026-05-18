# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
stratum colormap
====================

A colormap for displaying RGT values (created by Xinming Wu).

.. image:: ../../_static/cigvis/colormap/03.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/colormap/03.png'

import numpy as np
import cigvis

d = np.fromfile('../../data/rgt/ux.dat', np.float32).reshape(128, 128, 128)

nodes, cbar = cigvis.create_slices(d, [[36], [28], [84]],
                                   cmap='stratum',
                                   return_cbar=True,
                                   label_str='stratum cmap for RGT')
nodes.append(cbar)

cigvis.plot3D(nodes, save=cigvis.Plot3DSave(path='example.png'))
