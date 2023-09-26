# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
plot all custom cmaps
=======================

This example demonstrates how to plot all custom cmaps

.. image:: ../../_static/cigvis/colormap/02.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/colormap/02.png'

from cigvis import colormap

colormap.plot_all_custom_cmap([-5, 5], save='all_cmap.png', dpi=200)
