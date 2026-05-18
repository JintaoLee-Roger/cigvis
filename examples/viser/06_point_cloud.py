# Copyright (c) 2026 Jintao Li.
# All rights reserved.
"""
Point-cloud splats on the F3 demo
=================================

Use the same F3 scene as ``examples/3Dvispy/17-splat.py`` and render
interpreted objects with Viser's browser-native point clouds.

.. note::

    Viser point clouds are rendered and captured in the browser. Their apparent
    size depends on the browser viewport, device-pixel ratio, screenshot
    region, and ``point_shading`` setting, so a saved browser render may not
    match the interactive view pixel-for-pixel. Use the same viewport/camera
    and screenshot region when comparing images.


.. image:: ../../_static/cigvis/viser/06.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/viser/06.png'


from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

from cigvis import viserplot
from cigvis.io import load_skins


def _extend(nodes, new_nodes):
    if new_nodes:
        nodes.extend(new_nodes)


def salt_body_splats(salt, step=(5, 7, 4), max_points=30000):
    sample = np.asarray(salt[::step[0], ::step[1], ::step[2]]) > 0.0
    if not np.any(sample):
        return []

    eroded = binary_erosion(
        sample,
        structure=np.ones((3, 3, 3), dtype=bool),
        border_value=0,
    )
    boundary = sample & ~eroded
    pos = np.argwhere(boundary).astype(np.float32)
    pos *= np.asarray(step, dtype=np.float32)

    return viserplot.create_splats(
        pos,
        mode='surface',
        color=(0.0, 0.92, 0.95),
        size=2.0,
        max_points=max_points,
        seed=11,
    )


def surface_splats(surface, zmax, step=10, cmap='viridis', max_points=45000):
    ii = np.arange(0, surface.shape[0], step, dtype=np.float32)
    jj = np.arange(0, surface.shape[1], step, dtype=np.float32)
    grid_i, grid_j = np.meshgrid(ii, jj, indexing='ij')
    depth = np.asarray(surface[np.ix_(ii.astype(int), jj.astype(int))],
                       dtype=np.float32)

    valid = np.isfinite(depth) & (depth > 0.0) & (depth < zmax * 1.2)
    if not np.any(valid):
        return []

    pos = np.column_stack([grid_i[valid], grid_j[valid], depth[valid]])
    return viserplot.create_splats(
        pos.astype(np.float32),
        values=depth[valid],
        mode='surface',
        cmap=cmap,
        size=1.9,
        max_points=max_points,
        seed=13,
    )


def fault_skin_splats(skin_dir, max_points=35000):
    vertices, _faces, likelihood = load_skins(str(skin_dir),
                                              endian='>',
                                              values_type='likelihood')
    return viserplot.create_splats(
        vertices.astype(np.float32, copy=False),
        values=likelihood.astype(np.float32, copy=False),
        mode='surface',
        cmap='autumn',
        size=1.7,
        max_points=max_points,
        seed=17,
    )


def well_log_splats(log_path, zmax, sample_step=14):
    nlog = 4
    npoints = 2121
    x = np.asarray([259, 619, 339, 141], dtype=np.float32)
    y = np.asarray([33, 545, 704, 84], dtype=np.float32)
    z = np.arange(0, 0.2 * npoints, 0.2, dtype=np.float32)

    raw = np.fromfile(log_path, np.float32).reshape(nlog, npoints)
    with np.errstate(divide='ignore', invalid='ignore'):
        values = 0.5 * np.log(raw)

    all_pos = []
    all_values = []
    for i in range(nlog):
        valid = np.isfinite(values[i]) & (raw[i] > 0.0) & (z < zmax * 1.2)
        valid_idx = np.flatnonzero(valid)[::sample_step]
        if valid_idx.size == 0:
            continue
        all_pos.append(
            np.column_stack([
                np.full(valid_idx.size, x[i], dtype=np.float32),
                np.full(valid_idx.size, y[i], dtype=np.float32),
                z[valid_idx],
            ]))
        all_values.append(values[i, valid_idx])

    if not all_pos:
        return []

    pos = np.concatenate(all_pos).astype(np.float32)
    values = np.concatenate(all_values).astype(np.float32)
    return viserplot.create_splats(
        pos,
        values=values,
        mode='point',
        cmap='viridis',
        size=2.8,
    )


def pick_splats():
    pos = np.asarray([
        [192, 634.1855, 32.3816],
        [192, 616.5631, 139.5132],
        [192, 600.3925, 220.0604],
    ], dtype=np.float32)
    return viserplot.create_splats(
        pos,
        mode='point',
        color=(0.0, 0.95, 1.0),
        size=5.5,
    )


root = Path('/Volumes/T7/DATA/cigvisdata/F3/')
seisp = root / 'seis.dat'
saltp = root / 'salt.dat'
hz2p = root / 'hz.dat'
unc2p = root / 'unc2.dat'
logp = root / 'logs.dat'
skin_dir = root / 'skins'

ni, nx, nt = 591, 951, 362
shape = (ni, nx, nt)

seis = np.memmap(seisp, np.float32, 'c', shape=shape)
nodes = viserplot.create_slices(
    seis,
    pos=[ni - 2, 25, nt - 2],
    cmap='gray',
    clim=[-2.0, 1.5],
)

salt = np.memmap(saltp, np.float32, 'c', shape=shape)
_extend(nodes, salt_body_splats(salt))

hz2 = np.fromfile(hz2p, np.float32).reshape(ni, nx)
_extend(nodes, surface_splats(hz2, nt, step=10, cmap='viridis'))

unc2 = np.fromfile(unc2p, np.float32).reshape(ni, nx)
_extend(nodes, surface_splats(unc2, nt, step=12, cmap='cool'))

_extend(nodes, fault_skin_splats(skin_dir))
_extend(nodes, well_log_splats(logp, nt))
_extend(nodes, pick_splats())

viserplot.plot3D(nodes, 
                axis_scales=(1.0, 1.0, 1.7),
                fov=30.0,
                look_at=[0.529137, 0.942562, 0.48732],
                wxyz=[0.712576, 0.336748, 0.262983, 0.556486],
                position=[-1.485786, 1.445832, -1.21928],
                )
