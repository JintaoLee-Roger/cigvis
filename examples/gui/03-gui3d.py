# Copyright (c) 2024 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
GUI 3d demo
=================

Load a file and add some masks.

You can run the follow commands:

1. open a 3d gui panel
``python -c "import cigvis; cigvis.gui.gui3d()"``

2. open a 3d gui panel with special dimensions (128, 128, 128)
``python -c "import cigvis; cigvis.gui.gui3d(128, 128, 128)"``

3. open a 3d gui panel with special dimensions (128, 128, 128), and keep the dimensions when click the clear button.
``python -c "import cigvis; cigvis.gui.gui3d(128, 128, 128, False)"``


.. image:: ../../_static/cigvis/gui/03.gif
    :alt: image
    :align: center
"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/gui/03.png'

import cigvis

cigvis.gui.gui2d(128, 128, 128, False)
