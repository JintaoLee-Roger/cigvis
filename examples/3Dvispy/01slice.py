# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
显示一个3D数据体 (通过选取几个切片)
========================================

"""

import numpy as np
import cigvis

seisp = '../../data/co2/sx.dat'
ni, nx, nt = 192, 192, 240
sx = np.fromfile(seisp, np.float32).reshape(ni, nx, nt)

nodes, cbar = cigvis.create_slices(sx,
                                   cmap='Petrel',
                                   return_cbar=True,
                                   label_str='Amplitude')
nodes.append(cbar)

cigvis.plot3D(nodes, size=(1000, 800), savename='example.png')
