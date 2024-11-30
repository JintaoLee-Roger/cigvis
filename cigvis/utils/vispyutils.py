# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
utils for vispy visualization
"""

from typing import List
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


def valid_kwargs(ktype):
    if ktype == 'mesh':
        mesh_kwargs = [
            'vertices', 'faces', 'vertex_colors', 'face_colors', 'color',
            'vertex_values', 'meshdata', 'shading', 'mode', 'gcode', 'program',
            'vshare'
        ]
        return mesh_kwargs
    elif ktype == 'image':
        image_kwargs = [
            'data', 'method', 'grid', 'cmap', 'clim', 'gamma', 'interpolation',
            'texture_format', 'custom_kernel'
        ]
        return image_kwargs
    elif ktype == 'align_image':
        align_image_kwargs = [
            'image_funcs', 'axis', 'pos', 'limit', 'cmaps', 'clims',
            'interpolation', 'method'
        ]
        return align_image_kwargs
    elif ktype == 'colorbar':
        colorbar_kwargs = [
            'size', 'cmap', 'clim', 'discrete', 'disc_ticks', 'dpi_scale',
            'label_str', 'label_color', 'label_size', 'tick_size',
            'border_width', 'border_color', 'savedir', 'visible', 'parent'
        ]
        return colorbar_kwargs
    elif ktype == 'canvas':
        canvas_kwargs = [
            'title', 'size', 'position', 'show', 'autoswap', 'app',
            'create_native', 'vsync', 'resizable', 'decorate', 'fullscreen',
            'config', 'shared', 'keys', 'parent', 'dpi', 'always_on_top',
            'px_scale', 'bgcolor'
        ]
        return canvas_kwargs
    elif ktype == 'viscanvas':
        seiscanvas_kwargs = [
            'size', 'bgcolor', 'visual_nodes', 'grid', 'share',
            'cbar_region_ratio', 'scale_factor', 'center', 'fov', 'azimuth',
            'elevation', 'zoom_factor', 'axis_scales', 'auto_range', 'savedir',
            'title', 'dyn_light'
        ]
        return seiscanvas_kwargs
    elif ktype == 'line':
        line_kwargs = [
            'pos', 'color', 'width', 'connect', 'method', 'antialias'
        ]
        return line_kwargs
    else:
        raise ValueError(f"Invalid ktype: '{ktype}'. ktype can be one " +
                         "of 'mesh', 'image', 'align_image', 'colorbar', " +
                         "'canvas', 'seiscanvas', 'line'")


def get_valid_kwargs(ktype, **kwargs):
    valid = valid_kwargs(ktype)
    outkwargs = {key: value for key, value in kwargs.items() if key in valid}
    return outkwargs
