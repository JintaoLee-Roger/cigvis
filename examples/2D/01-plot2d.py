# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Show 2d slices
================

cbar can be set to ``cbar=''``, which ignores cbar_label.

.. image:: ../../_static/cigvis/2D/01.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/2D/01.png'

import cigvis
import numpy as np

d = np.fromfile('../../data/rgt/sx.dat', np.float32).reshape(128, 128, 128)
sl = d[30, :, :]

cigvis.plot2d(sl,
              figsize=(6, 6),
              xlabel='xline',
              ylabel='time',
              title='Example',
              cbar='Amplitude',
              save='example.png',
              dpi=200)
