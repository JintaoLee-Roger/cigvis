# Copyright (c) 2026 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Automatic PNG export
====================

Save a PNG without pressing ``s`` in the VisPy window. PNG export uses the
visible canvas framebuffer and a transparent background by default; set
``transparent_bg=False`` when a solid background is needed.
"""

import numpy as np
import cigvis
from pathlib import Path


root = Path(__file__).resolve().parent.parent.parent

root = Path('/Volumes/T7/DATA/cigvisdata/')
sxp = root / 'rgt3d/seis.dat'
ni, nx, nt = 128, 128, 128
sx = np.fromfile(sxp, np.float32).reshape(ni, nx, nt)

nodes = cigvis.create_slices(sx, pos=[[20], [40], [100]], cmap='Petrel')
nodes += cigvis.create_colorbar_from_nodes(nodes, 'Amplitude', select='slices')

cigvis.plot3D(
    nodes,
    view=cigvis.Plot3DView(
        size=(700, 600),
    ),
    save=cigvis.Plot3DSave(
        path='example.png',
    ),
    run_app=False,
)


