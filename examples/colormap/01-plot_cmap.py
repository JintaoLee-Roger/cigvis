# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
plot colormap
================

This example demonstrates how to plot a colormap

.. image:: ../../_static/cigvis/colormap/01.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/colormap/01.png'

from cigvis import colormap

colormap.plot_cmap('Petrel', [-5, 5], save='cmap.png')