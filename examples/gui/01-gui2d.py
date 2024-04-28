# Copyright (c) 2024 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
GUI 2d demo
=================

Load a file and add some masks.

You can run the follow commands:

1. open a 2d gui panel
``python -c "import cigvis; cigvis.gui.gui2d()"``

2. open a 2d gui panel with special dimensions (512, 512)
``python -c "import cigvis; cigvis.gui.gui2d(512, 512)"``

3. open a 2d gui panel with special dimensions (512, 512), and keep the dimensions when click the clear button.
``python -c "import cigvis; cigvis.gui.gui2d(512, 512, False)"``


.. image:: ../../_static/cigvis/gui/01.gif
    :alt: image
    :align: center
"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/gui/01.png'

import cigvis

cigvis.gui.gui2d(512, 512, False)
