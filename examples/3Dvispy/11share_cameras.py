# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
在多个canvas中, 共享相机参数
================================================

这个可以很方便的实现比较两个数据结果
"""

import numpy as np
import cigvis

seisp = '../../data/co2/sx.dat'
ni, nx, nt = 192, 192, 240
sx = np.fromfile(seisp, np.float32).reshape(ni, nx, nt)

nodes1 = cigvis.create_slices(sx, cmap='Petrel')
nodes2 = cigvis.create_slices(sx, cmap='Petrel')

cigvis.plot3D(
    [nodes1, nodes2],
    grid=(1, 2),  # here, define a grid
    share=True,  # here, link all cameras
    size=(1000, 800),
    savename='example.png')
