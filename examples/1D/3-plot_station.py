# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
seismogram
================

Plot seismogram offset with stations

.. image:: ../../_static/cigvis/1D/3.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/1D/3.png'

import numpy as np
import cigvis
import matplotlib.pyplot as plt

data = np.load('../../data/smg/data.npy')
offset = np.load('../../data/smg/offset.npy')
offset_index = np.load('../../data/smg/offset_index.npy')

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

cigvis.plot_signal_compare(data, ax=axes[0])
cigvis.plot_signal_compare(data / 2,
                           offset,
                           offset_index,
                           with_offset=True,
                           ax=axes[1])

plt.tight_layout()
plt.savefig('3.png', bbox_inches='tight', pad_inches=0.01, dpi=200)
plt.show()