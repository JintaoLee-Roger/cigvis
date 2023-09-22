# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
在3D地震数据体的切片上叠加多个3D数据体 (RGT, fault)
===================================================

``create_overlay`` 的第一个参数是背景(background), 
第二个参数是叠加的前景(foreground), 其可以包含多个3D数据体

.. Note::
    foreground 需要合理设置透明度和mask

"""

import numpy as np
import cigvis
from cigvis import colormap

sxp = '../../data/rgt/sx.dat'
uxp = '../../data/rgt/ux.dat'
fxp = '../../data/rgt/fx.dat'
ni, nx, nt = 128, 128, 128

sx = np.fromfile(sxp, np.float32).reshape(ni, nx, nt)
rgt = np.fromfile(uxp, np.float32).reshape(ni, nx, nt)
fx = np.fromfile(fxp, np.float32).reshape(ni, nx, nt)

rgt_cmap = colormap.set_alpha('jet', 0.4)
# mask min value (0), 0 means no fault
fx_cmap = colormap.set_alpha_except_min('jet', alpha=1)

fg_cmap = [rgt_cmap, fx_cmap]

# fx is discrete data, set interpolation as 'nearest'
nodes = cigvis.create_overlay(sx, [rgt, fx],
                              pos=[[36], [28], [84]],
                              bg_cmap='gray',
                              fg_cmap=fg_cmap,
                              fg_interpolation=['cubic', 'nearest'])

cigvis.plot3D(nodes, size=(800, 800), savename='example.png')
