"""
Thin-volume slice viewer
========================

``sliceviewer`` is useful when a remote server cannot render a heavy 3D
scene but a 2D plane through a 3D/4D array is enough. ``display_axes`` selects
the two dimensions rendered as the image; all other dimensions become numeric
index controls in the sidebar.
"""

import numpy as np

from cigvis import sliceviewer as sv


rng = np.random.default_rng(4)
volume = rng.normal(size=(4, 64, 96)).astype(np.float32)
volume += np.linspace(-1, 1, volume.shape[2], dtype=np.float32)[None, None, :]

mask = np.zeros_like(volume)
mask[:, 22:42, 34:70] = 1

nodes = sv.create_slice(volume, display_axes=(1, 2), indices={0: 2}, cmap="gray")
nodes = sv.add_mask(nodes, mask, cmaps="jet", alpha=0.45, excpt="min")

x = np.arange(volume.shape[2])
y = 34 + 8 * np.sin(x / 12)
nodes += sv.add_horizon(x, y, name="horizon")


if __name__ == "__main__":
    # On a remote server, forward this port with SSH and open it locally.
    # Example: ssh -L 5007:localhost:5007 user@server
    sv.show(nodes, port=5007, title="Thin volume")
