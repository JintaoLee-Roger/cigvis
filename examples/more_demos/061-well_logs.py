"""
Log trajectories and log curves
=========================================

.. image:: ../../_static/cigvis/more_demos/061.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/more_demos/061.png'

import numpy as np
import cigvis
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent


def show(sx, points, values, null_value):
    """
    Display the well log as a constant-radius tube colored by one log curve.
    """

    v = values[:, 0]

    v2 = v.copy()
    v2[v2 < -900] = v[v!=-999.25].min()
    logs1 = np.concatenate([points, v2[:, np.newaxis]], axis=1)
    nodes0 = cigvis.create_slices(sx)
    nodes0 += cigvis.create_line_logs(
        logs1[:, :4],
        value_type='amp')

    nodes1 = cigvis.create_slices(sx)
    nodes1 += cigvis.create_well_logs(points,
                                      v,
                                      cmap='jet',
                                      radius_tube=2,
                                      null_value=null_value)
    """
    Display the well log as a variable-radius tube colored by one log curve.
    The radius also encodes the log curve value.
    """

    nodes2 = cigvis.create_slices(sx)
    nodes2 += cigvis.create_well_logs(points,
                                      v,
                                      cmap='jet',
                                      cyclinder=False,
                                      radius_tube=[1, 2],
                                      null_value=null_value)
    """
    Display multiple log curves.

    Display the well log as a variable-radius tube colored by the first log
    curve. The radius also encodes the first log curve value.

    Other log curves are displayed as faces attached to the tube surface.
    """

    cmaps = ['jet', 'seismic', 'Petrel', 'od_seismic1']

    nodes3 = cigvis.create_slices(sx)
    nodes3 += cigvis.create_well_logs(points,
                                      values,
                                      cmap=cmaps,
                                      cyclinder=False,
                                      radius_tube=[1, 1.7],
                                      radius_line=[2.2, 5],
                                      null_value=null_value)

    cigvis.plot3D(
        [nodes0, nodes1, nodes2, nodes3],
        view=cigvis.Plot3DView(
            grid=(2, 2),
            # zoom_factor=16,
            share=True,
        ),
        save=cigvis.Plot3DSave(path='example.png'))


if __name__ == '__main__':
    sxp = root / 'data/co2/sx.dat'
    lxp = root / 'data/co2/lx.dat'
    lasp = root / 'data/cb23.las'
    ni, nx, nt = 192, 192, 240

    sx = np.memmap(sxp, np.float32, 'c', shape=(ni, nx, nt))
    lx = np.memmap(lxp, np.float32, 'c', shape=(ni, nx, nt))

    # create a well log
    las = cigvis.io.load_las(lasp)
    idx = las['Well']['name'].index('NULL')
    null_value = float(las['Well']['value'][3])
    lasdata = las['data'][:, 1:]

    x = np.linspace(50, 100, len(lasdata))
    y = np.linspace(50, 150, len(lasdata))
    z = np.sin((y - 50) / 200 * np.pi) * 200
    points = np.c_[x, y, z]

    show(sx, points, lasdata[:, 1:5], null_value)
