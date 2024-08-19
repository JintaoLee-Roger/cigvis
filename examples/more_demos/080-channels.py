# Copyright (c) 2024 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Channels demo
===============

.. image:: ../../_static/cigvis/more_demos/080.png
    :alt: image
    :align: center
"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/more_demos/080.png'

import numpy as np
import cigvis
from cigvis import colormap
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
root = root / 'data/channel'

n1, n2, n3 = 256, 256, 256
shape = (n1, n2, n3)

seis = np.memmap(root / 'Seismic_6.dat', np.float32, mode='c', shape=shape)
labl = np.memmap(root / 'Label_6.dat', np.uint8, mode='c', shape=shape)
iped = np.memmap(root / 'Ip_6.dat', np.float32, mode='c', shape=shape)

pos = [6, 6, 250]

vis1 = cigvis.create_slices(seis, pos=pos)
vis2 = cigvis.create_slices(iped, cmap='jet', pos=pos)
fg_cmap = colormap.set_alpha_except_min('jet', 1)
# vis3 = cigvis.create_overlay(seis,
#                              labl,
#                              pos=pos,
#                              bg_cmap='gray',
#                              fg_cmap=fg_cmap,
#                              fg_interpolation='nearest')
vis3 = cigvis.create_slices(seis, pos=pos, cmap='gray')
vis3 = cigvis.add_mask(vis3, labl, cmaps=fg_cmap, interpolation='nearest')

# vis4 = cigvis.create_overlay(seis,
#                              labl,
#                              pos=pos,
#                              bg_cmap='Petrel',
#                              fg_cmap=fg_cmap,
#                              fg_interpolation='nearest')
vis4 = cigvis.create_slices(seis, pos=pos, cmap='Petrel')
vis4 = cigvis.add_mask(vis4, labl, cmaps=fg_cmap, interpolation='nearest')

vis4 += cigvis.create_bodys(labl, 0.5, 0)

cigvis.plot3D([vis1, vis2, vis3, vis4], (2, 2), True, False, size=(1200, 1100))
