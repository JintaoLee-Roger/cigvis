"""
CigVis GUI — modern PySide6-based viewer.

Provides:
  gui2d : 2D viewer (matplotlib) with collapsible sidebar
           - supports 3D data as collection of 2D slices
           - SAM-like point/box/brush annotation
  gui3d : 3D viewer (vispy) with collapsible sidebar
           - uses VolumeImage (faster overlay management)
           - optional SAM-like interactive segmentation

Quick start::

    from cigvis.gui import gui2d, gui3d

    # 2D viewer
    gui2d(nx=256, ny=512)

    # 3D viewer
    gui3d(nx=128, ny=128, nz=512)

    # 3D viewer with pre-loaded data
    import numpy as np
    data = np.fromfile("volume.dat", np.float32).reshape(128, 128, 512)
    gui3d(data=data)

    # 3D viewer with SAM inference
    def my_decode(base_vol, prompts, xyz):
        # ... run your model ...
        return mask   # ndarray, same shape as base_vol

    gui3d(data=data, decode_fn=my_decode)
"""

from .gui2d import gui2d, Gui2dWindow
from .gui3d import gui3d, Gui3dWindow

__all__ = ['gui2d', 'Gui2dWindow', 'gui3d', 'Gui3dWindow']
