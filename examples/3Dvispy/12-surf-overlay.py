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
from skimage.filters import gaussian

root = Path(__file__).resolve().parent.parent.parent
root = root / 'data/channel/'

n1, n2, n3 = 256, 256, 256
shape = (n1, n2, n3)

# seismic
seis = np.memmap(root / 'Seismic_6.dat', np.float32, mode='c', shape=shape)
# channel label
labl = np.memmap(root / 'Label_6.dat', np.uint8, mode='c', shape=shape)
# surface
sf = np.fromfile(root / 'mh2.dat', np.float32).reshape(n1, n2)

# amplitude of channel's label
labl_sf = cigvis.utils.surfaceutils.interp_surf(labl, sf, order=1)

# labl_sf only contains 0 and 1, smooth the labl_sf to simulate the probability, this is not necessary
labl_sf = gaussian(labl_sf, sigma=10)
labl_sf = labl_sf / labl_sf.max()

# set a colormap for foreground, i.e., label
labl_cmap = colormap.set_alpha_except_min('jet', 0.6, False)

nodes = cigvis.create_slices(seis, cmap='Petrel')
nodes += cigvis.create_surfaces(
    sf,
    seis,
    value_type=['amp', labl_sf], # two value types
    cmap=['Petrel', labl_cmap], # first is for seismic, second is for label
    clim=[[seis.min(), seis.max()], [0, 1]], # clim for two value types
)  # set as 'value'

# idx=0 means select the first Surface node of the nodes (though there is only one Surface node in this case).
# idx2=1 means select the second cmap, and clim for the select Surface node, (`labl_cmap` in this case).
nodes += cigvis.create_colorbar_from_nodes(nodes, 'surface', select='surface', idx=0, idx2=1)

cigvis.plot3D(nodes, xyz_axis=False)
