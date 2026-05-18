# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
utils for vispy visualization
"""

import numpy as np


def set_canvas_size(size, ratio, append=False):
    """
    set canvas size and colorbar size
    """
    if append:
        size = (size[0] / (1 - ratio), size[1])

    cbar_size = (size[0] * ratio, size[1])

    return cbar_size, size


def init_cbar_region_ratio(cbar):
    """
    init colorbar region ratio
    """
    cbar_label = cbar.label_str
    clim = cbar.clim_
    cbar_region_ratio = 0.1
    if cbar_label is not None and cbar_label != '':
        cbar_region_ratio += 0.025
    if np.abs(clim).max() >= 10:
        cbar_region_ratio += 0.01
    if np.abs(clim).max() >= 100:
        cbar_region_ratio += 0.01
    if np.abs(clim).max() >= 1000:
        cbar_region_ratio += 0.01
    if np.abs(clim).max() >= 10000:
        cbar_region_ratio += 0.01
    if np.abs(clim).max() <= 1:
        cbar_region_ratio += 0.03

    return cbar_region_ratio
