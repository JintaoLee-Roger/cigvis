# Copyright (c) 2025 Jintao Li. 
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
3D RGB/RGBA volume, i.e., 4D array
========================================

Display a 3D RGB/RGBA volume, whose shape could be (ni, nx, nt, 3/4) or (3/4, ni, nx, nt).

This would be useful for analyzing the inner features of a deep 3D network (using PCA to keep the most important three features).

.. image:: ../../_static/cigvis/3Dvispy/15.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/3Dvispy/15.png'

import numpy as np
import cigvis 

from pathlib import Path
root = Path(__file__).resolve().parent.parent.parent

seisp = root / 'data/co2/sx.dat'
ni, nx, nt = 192, 192, 240
data = np.fromfile(seisp, np.float32).reshape(ni, nx, nt)
data = (data - data.min()) / (data.max() - data.min())

### FP32, RGBA volume's range should be in [0, 1]
sx = np.zeros((ni, nx, nt, 4)).astype(np.float32)
sx[:, :, :, 0] = data 
sx[:, :, :, 1] = data + 0.2
sx[:, :, :, 2] = data - 0.2
sx[:, :, :, 3] = np.random.randn(ni, nx, nt)
sx = np.clip(sx, 0, 1)

### UINT8, RGBA volume's range should be in [0, 255]
sx = (sx * 255).astype(np.uint8)

### `cmap` will be ignored
nodes = cigvis.create_slices(sx, pos=[[10, ni//2, ni-10], [10, nx//2, nx-10], [10, nt//2, nt-10]])

cigvis.plot3D(nodes, size=(700, 600))
