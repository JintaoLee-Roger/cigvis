"""
2D & 1D
======================

.. image:: ../../_static/cigvis/2D/13.png
    :alt: image
    :align: center

"""

import numpy as np
import cigvis
from cigvis import colormap
import matplotlib.pyplot as plt

root = '/Users/lijintao/Downloads/data/F3/'
seisp = root + 'seis.dat'
saltp = root + 'salt.dat'
hz2p = root + 'hz.dat'
unc1p = root + 'unc1.dat'
unc2p = root + 'unc2.dat'
logsp = root + 'logs.dat'
ni, nx, nt = 591, 951, 362
shape = (ni, nx, nt)

seis = np.memmap(seisp, np.float32, 'c', shape=shape)
# overlay
inter = np.memmap(root + 'overlay.dat', np.float32, 'c', shape=shape)

salt = np.memmap(saltp, np.float32, 'c', shape=shape)
hz2 = np.fromfile(hz2p, np.float32).reshape(ni, nx)
unc = np.memmap(root + 'unc.dat', np.float32).reshape(shape)
unc2 = np.fromfile(unc2p, np.float32).reshape(ni, nx).astype(np.float32)
logs = np.fromfile(logsp, np.float32).reshape(4, 2121)

sx2 = seis[259, :, :]
ov2 = inter[259, :, :]
sl2 = salt[259, :, :]
sl2[sl2 > 0] = 1
sl2[sl2 <= 0] = 0
hzl = hz2[259, :]
unl = unc2[259, :]

w = 80
logz = np.arange(0, 0.2 * 2121, 0.2)
logv = logs[0][logz < nt - 1]
logv = (logv - logv.min()) * w / np.abs(logv).max()
logx = np.array([33] * 2121)[logz < nt - 1]
logz = logz[logz < nt - 1]
logxv = logx + logv - 5

fig, axs = plt.subplots(2, 1, figsize=(8, 9))

fg1 = {}
fg1['img'] = cigvis.fg_image_args(
    ov2,
    alpha=0.5,
    clim=[inter.max() * 0.15, inter.max() * 0.5],
    interpolation='nearest',
)

cmap = colormap.custom_disc_cmap([0, 1], ['red', 'cyan'])
cmap = colormap.set_alpha_except_min(cmap, 1, False)
fg1['img'] += cigvis.fg_image_args(
    sl2,
    cmap=cmap,
    interpolation='nearest',
)

fg1['line'] = cigvis.line_args(np.arange(nx), hzl, 'blue')
fg1['line'] += cigvis.line_args(np.arange(nx), unl, 'white')
fg1['line'] += cigvis.line_args(logxv, logz, 'black', lw=0.6)

cigvis.plot2d(
    sx2,
    fg=fg1,
    aspect='auto',
    xsample=[0, 0.0125],
    ysample=[0, 0.002],
    xlabel='Crossline / km',
    ylabel='Time / s',
    title='2D section (Inline=259)',
    ax=axs[0],
)

traces = sx2[:50, :].T

cigvis.plot_multi_traces(
    traces,
    dt=0.002,
    c='black',
    fill_up=0.2,
    ax=axs[1],
)

plt.tight_layout()
plt.savefig('2Dcanvas.png', bbox_inches='tight', pad_inches=0.01, dpi=300)

plt.show()
