import numpy as np
from cigvis import colormap, viserplot

data = np.fromfile('data/rgt/sx.dat', dtype=np.float32).reshape(128, 128, 128)
rgt = np.fromfile('data/rgt/ux.dat', dtype=np.float32).reshape(128, 128, 128)
fx = np.fromfile('data/rgt/fx.dat', dtype=np.float32).reshape(128, 128, 128)
nodes = viserplot.create_slices(data, cmap='gray')

cmaps = [
    colormap.set_alpha('stratum', 0.5, False),
    colormap.set_alpha_except_min('jet', 1, False)
]
clims = [[rgt.min(), rgt.max()], [fx.min(), fx.max()]]
nodes = viserplot.add_mask(nodes, [rgt, fx], clims, cmaps)

viserplot.plot3D(nodes)
