# Copyright (c) 2024 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Overlaying channels on a horizon display
==========================================

.. image:: ../../_static/cigvis/3Dvispy/12.png
    :alt: image
    :align: center
"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/3Dvispy/12.png'


import numpy as np
import cigvis
from cigvis import colormap
from pathlib import Path

root = Path('/Users/lijintao/work/mygit/cigvis/data/channel/')

n1, n2, n3 = 256, 256, 256
shape = (n1, n2, n3)

# seismic
seis = np.memmap(root / 'Seismic_6.dat', np.float32, mode='c', shape=shape)
# channel label
labl = np.memmap(root / 'Label_6.dat', np.uint8, mode='c', shape=shape)
# surface
sf = np.fromfile(root / 'mh2.dat', np.float32).reshape(n1, n2)

# amplitude of seismic
seis_sf = cigvis.utils.surfaceutils.interp_surf(seis, sf)
# amplitude of channel's label
labl_sf = cigvis.utils.surfaceutils.interp_surf(labl, sf, order=1)
# set a colormap for foreground, i.e., label
labl_cmap = colormap.set_alpha_except_min('jet', 0.5, False)
# blending two array into a color array, whose shape is (ni, nx, 3)
colors = colormap.blend_two_arrays(seis_sf, labl_sf, 'Petrel', labl_cmap,
                                   [seis.min(), seis.max()], [0, 1])
# concatenate the surface pos and the correspanding colors, (ni, nx, 4)
sfc = np.concatenate([sf[:, :, np.newaxis], colors], axis=2)

nodes = cigvis.create_slices(seis, cmap='Petrel')
nodes += cigvis.create_surfaces(sfc, value_type='value') # set as 'value'

cigvis.plot3D(nodes, xyz_axis=False)
