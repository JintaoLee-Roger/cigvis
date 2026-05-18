"""
Real seismic slice viewer
=========================

``sliceviewer`` is useful when a remote server cannot render a heavy 3D
scene but a 2D plane through a 3D/4D array is enough. ``display_axes`` selects
the two dimensions rendered as the image; all other dimensions become numeric
index controls in the sidebar.


.. image:: ../../_static/cigvis/sliceviewer/01.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/sliceviewer/01.png'


from pathlib import Path

import numpy as np

from cigvis import sliceviewer as sv


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "rgt"
SHAPE = (128, 128, 128)
INLINE = 40

seismic = np.fromfile(DATA / "sx.dat", np.float32).reshape(SHAPE)
fault = np.fromfile(DATA / "fx.dat", np.float32).reshape(SHAPE)
rgt = np.fromfile(DATA / "ux.dat", np.float32).reshape(SHAPE)

nodes = sv.create_slice(
    seismic,
    display_axes=(2, 1),
    indices={0: INLINE},
    axis_labels=("inline", "crossline", "time"),
    cmap="gray",
    interpolation="nearest",
    render_mode="float",
)
nodes = sv.add_mask(nodes, fault, cmap="jet", alpha=0.45, excpt="min")

rgt_section = rgt[INLINE]
target = float(np.nanpercentile(rgt_section, 55))
time = np.nanargmin(np.abs(rgt_section - target), axis=1)
crossline = np.arange(SHAPE[1])
nodes += sv.add_horizon(
    crossline,
    time,
    name=f"RGT {target:.2f}",
    color="yellow",
    axes=(1, 2),
)


if __name__ == "__main__":
    # On a remote server, forward this port with SSH and open it locally.
    # Example: ssh -L 5007:localhost:5007 user@server
    sv.show(
        nodes,
        port=5007,
        title="RGT seismic slice",
        plot_height=520,
        xlim=(0, SHAPE[1] - 1),
        ylim=(0, SHAPE[2] - 1),
    )
