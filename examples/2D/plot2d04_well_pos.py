# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
显示2d的 水平 slice 和 测井的位置
==================================

"""

import cigvis
import numpy as np
from cigvis import colormap

root = '../../'
sxp = 'data/rgt/sx.dat'
ni, nx, nt = 128, 128, 128

sx = np.memmap(root + sxp, np.float32, 'r', shape=(ni, nx, nt))
sx2 = sx[:, :, 40]

fg = {}

well_pos = np.random.rand(10, 2) * 100 + 10
well_name = ['well-' + str(i + 10001) for i in range(10)]
fg['marker'] = cigvis.marker_args(well_pos[:, 0],
                                  well_pos[:, 1],
                                  marker='^',
                                  c='red')
fg['annotate'] = cigvis.annotate_args(well_pos[:, 0], well_pos[:, 1],
                                      well_name)

cigvis.plot2d(sx2,
              fg,
              figsize=(6, 6),
              xlabel='inline',
              ylabel='crossline',
              title='Example',
              save='example.png',
              dpi=200)
