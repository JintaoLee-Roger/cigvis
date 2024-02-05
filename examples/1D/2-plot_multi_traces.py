# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
显示多条曲线
==========================

.. image:: ../../_static/cigvis/1D/2.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/1D/2.png'

import numpy as np
import cigvis
import matplotlib.pyplot as plt

root = '../../data/'
sxp = root + 'seis_h360x600x400.dat'
ni, nx, nt = 400, 600, 360

sx = np.fromfile(sxp, np.float32).reshape(ni, nx, nt)

traces = sx[100, 50:70, :].T

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

cigvis.plot_multi_traces(traces, dt=0.02, c='black', ax=axes[0])

# fill mode
cigvis.plot_multi_traces(traces, dt=0.02, c='black', fill_up=0.2, ax=axes[1])

plt.tight_layout()
plt.savefig('example.png', bbox_inches='tight', pad_inches=0.01, dpi=200)
plt.show()
