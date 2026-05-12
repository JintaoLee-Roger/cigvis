# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Overlaying fault displays on slices of 3D seismic data bodies
===================================================================

``create_overlay``: the first parameters is (background), 
and the second parameters is (foreground)

.. Note::
    Set foreground transparency and masking carefully.

.. image:: ../../_static/cigvis/3Dvispy/02.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/3Dvispy/02.png'

import numpy as np
import cigvis
from cigvis import colormap
from pathlib import Path
from cigvis.vispynodes.splat import Splat
root = Path(__file__).resolve().parent.parent.parent

sxp = root / 'data/rgt/sx.dat'
fxp = root / 'data/rgt/fx.dat'
ni, nx, nt = 128, 128, 128

sx = np.fromfile(sxp, np.float32).reshape(ni, nx, nt)
fx = np.fromfile(fxp, np.float32).reshape(ni, nx, nt)

coords = np.argwhere(fx > 0).astype(np.float32)

splat = Splat(scaling="visual", sigma_rel=0.55, cutoff=1e-3, alpha=0.8)
splat.set_data(pos=coords, size=2.0, color=(1, 0.8, 0.2, 1.0))

# mask min value (0), 0 means no fault
fg_cmap = colormap.set_alpha_except_min('jet', alpha=1)

# fx is discrete data, set interpolation as 'nearest'
nodes = cigvis.create_slices(sx, pos=[[36], [28], [84]], cmap='gray')
nodes = cigvis.add_mask(nodes, fx, cmaps=fg_cmap, interpolation='nearest')
nodes += cigvis.create_colorbar_from_nodes(nodes, 'Amplitude', select='slices')
nodes += [splat]

cigvis.plot3D(
    nodes,
    view=cigvis.Plot3DView(size=(700, 600)),
    save=cigvis.Plot3DSave(path='example.png'),
)
