# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
在3D地震数据体的切片上叠加RGT显示
=================================================

``create_overlay`` 的第一个参数是背景(background), 
第二个参数是叠加的前景(foreground)

.. Note::
    foreground 需要合理设置透明度和mask

"""

import numpy as np
import cigvis
from cigvis import colormap

sxp = '../../data/rgt/sx.dat'
uxp = '../../data/rgt/ux.dat'
ni, nx, nt = 128, 128, 128

sx = np.fromfile(sxp, np.float32).reshape(ni, nx, nt)
rgt = np.fromfile(uxp, np.float32).reshape(ni, nx, nt)

fg_cmap = colormap.set_alpha('jet', alpha=0.4)
nodes = cigvis.create_overlay(sx,
                              rgt,
                              pos=[[36], [28], [84]],
                              bg_cmap='gray',
                              fg_cmap=fg_cmap)

cigvis.plot3D(nodes, size=(800, 800), savename='example.png')
