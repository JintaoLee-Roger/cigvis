# Copyright (c) 2024 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Display axis
============================================================

.. image:: ../../_static/cigvis/3Dvispy/13.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/3Dvispy/13.png'

import numpy as np
import cigvis
from pathlib import Path
root = Path(__file__).resolve().parent.parent.parent

seisp = root / 'data/co2/sx.dat'
ni, nx, nt = 192, 192, 240
sx = np.fromfile(seisp, np.float32).reshape(ni, nx, nt)

nodes1 = cigvis.create_slices(sx, pos=[[0, 191], [0, 191], [0, 239]])
# Add a north pointer via passing north_direction, if axis_pos is 'auto', axis will change when rotating
nodes1 += cigvis.create_axis(sx.shape, 'box', axis_pos='auto', north_direction=[0, 1])

nodes2 = cigvis.create_slices(sx, pos=[[0, 191], [0, 191], [0, 239]])
# Use intervals and starts to control the tick labels, and set axis_labels to show the axis labels
nodes2 += cigvis.create_axis(sx.shape, 'axis', axis_pos='auto', intervals=[0.025, 0.025, 0.004], starts=[10, 10, 3], axis_labels=['Inline/km', 'Crossline/km', 'Time/s'])

nodes3 = cigvis.create_slices(sx, pos=[[0, 191], [0, 191], [0, 239]])
# if want to show the axis in the specific position, set axis_pos, and the axis will be fixed
# there are some other parameters to control the axis, such as tick_nums, ticks_font_size, labels_font_size, ticks_length
nodes3 += cigvis.create_axis(sx.shape, 'axis', axis_pos=[0, 0, 1], tick_nums=4, ticks_font_size=26, labels_font_size=30, ticks_length=6)

cigvis.plot3D([nodes1, nodes2, nodes3],
              xyz_axis=False,
              grid=(1, 3),
              share=True,
              size=(1600, 500),
              savename='example.png')
