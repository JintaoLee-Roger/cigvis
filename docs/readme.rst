.. image:: https://raw.githubusercontent.com/JintaoLee-Roger/images/main/cigvis/more_demos/070.png
   :align: center
   :alt: 01

CIGVis - a tool for visualizing multidimensional geophysical data
======================================================================

Overview
------------

**cigvis** is a tool for visualizing multidimensional geophysical data, 
developed by the 
`Computational Interpretation Group (CIG) <https://cig.ustc.edu.cn/main.htm>`_. 
Users can quickly visualize data with just a few lines of code.

cigvis can be used for various geophysical data visualizations, 
including 3D seismic data, overlays of seismic data with other 
information like labels, faults, RGT, horizon surfaces, well log 
trajectories, and well log curves, 3D geological bodies, 2D data, 
and 1D data, among others. Its GitHub repository can be found at 
`github.com/JintaoLee-Roger/cigvis <https://github.com/JintaoLee-Roger/cigvis>`_, 
and documentation is available at 
`https://cigvis.readthedocs.io/ <https://cigvis.readthedocs.io/>`_.

cigvis leverages the power of underlying libraries such as 
`vispy <https://github.com/vispy/vispy>`_ for 3D visualization, 
`matplotlib <https://matplotlib.org/>`_ for 2D and 1D visualization, 
and `plotly <https://plotly.com/>`_ for Jupyter environments (work in progress). 
The 3D visualization component is heavily based on the code from 
`yunzhishi/seismic-canvas <https://github.com/yunzhishi/seismic-canvas>`_ 
and has been further developed upon this foundation.

Installation
----------------

To install via PyPI, use

.. code-block:: bash

    # Minimal installation
    pip install cigvis

    # include plotly 
    pip install "cigvis[plotly]"

    # include viser
    pip install "cigvis[viser]"

    # install all dependencies
    pip install "cigvis[all]"



For local installation, clone the repository from GitHub and then install it using pip:

.. code-block:: bash

    git clone https://github.com/JintaoLee-Roger/cigvis.git

    # Minimal installation
    pip install -e .

    # include plotly 
    pip install -e ".[plotly]"

    # include viser
    pip install -e ".[viser]"

    # install all dependencies
    pip install -e ".[all]"



Core Features
-----------------

1. Convenient 3D geophysical data visualization.
2. Ongoing development of 2D and 1D data visualization.
3. Additional colormaps tailored for geophysical data.
4. Rapid display of large data in OpenVDS format.

Usage
---------

Basic Structure
^^^^^^^^^^^^^^^^^^

The fundamental structure of cigvis's visualization code consists of:
1. Data loading
2. Creating nodes
3. Passing nodes to the ``plot3D`` function

For example

.. code-block:: bash

    import numpy as np
    import cigvis

    # Load data
    d = np.fromfile('sx.dat', np.float32).reshape(ni, nt, nx)

    # Create nodes
    nodes = cigvis.create_slices(d)

    # Visualize in 3D
    cigvis.plot3D(nodes)


This basic code structure allows you to quickly visualize your geophysical data using cigvis. Simply load your data, create nodes, and pass them to the ``plot3D`` function as demonstrated in the example above.

Camera and Dragging
^^^^^^^^^^^^^^^^^^^^^^

Left click and drag to rotate the camera angle; right click and drag, or scroll mouse wheel, to zoom in and out. Hold ``<Shift>`` key, left click and drag to pan move. Press ``<Space>`` key to return to the initial view. Press ``<S>`` key to save a screenshot PNG file at any time. Press ``<Esc>`` key to close the window.

Hold ``<Ctrl>`` key, the selectable visual nodes will be highlighted when your mouse hovers over them; left click and drag to move the highlighted visual node. The volume slices will update their contents in real-time during dragging. You can also press ``<D>`` key to toggle the dragging mode on/off.

Press ``<z>`` to zoom in z axis, press ``<Z>`` or ``<Shift> + <z>`` to zoom out z axis. 
Press ``<f>`` to increase ``fov`` value, press ``<F>`` or ``<Shift> + <f>`` to decrease ``fov`` value.

.. image:: https://raw.githubusercontent.com/JintaoLee-Roger/images/main/cigvis/ex.gif
   :alt: ex1
   :align: center

Press ``<s>`` to save a screen shot.

Press ``<a>`` to print the camera's parameters in real-time; hold on 
the ``<alt>`` (or ``<option>`` in macos) and left click the mouse to 
show the coordinate of the click point in the 3D volume.

.. image:: https://raw.githubusercontent.com/JintaoLee-Roger/images/main/cigvis/ex2.gif
   :alt: ex2
   :align: center

Various Geophysical data
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In cigvis, we represent various geophysical data as individual nodes, assemble these nodes into a list, and then pass this list to the ``plot3D`` function for visualization.

We visualize a three-dimensional data volume as multiple slices along the x, y, and z directions. Additionally, we can overlay other three-dimensional data slices on these slices, allowing users to interactively drag them along an axis using the mouse.

Horizon data can be represented as scatter points with a shape of (N, 3), or as z-values on a regular grid of size (n1, n2).

Well log trajectories are displayed as tubes, where the size of the first well log curve is represented by the color and radius at each position along the tube. Other well log curves are displayed as surfaces attached to the tube's edge. An example is shown below 
(code available at `cigvis/gallery/3Dvispy/09 <https://cigvis.readthedocs.io/en/latest/gallery/3Dvispy/09-slice_surf_body_logs.html#sphx-glr-gallery-3dvispy-09-slice-surf-body-logs-py>`_).

.. image:: https://raw.githubusercontent.com/JintaoLee-Roger/images/main/cigvis/3Dvispy/09.png
   :alt: 09
   :align: center

These capabilities within cigvis allow for versatile and interactive visualizations of a wide range of geophysical data types, enhancing the understanding and analysis of such data in geoscience applications.

Multivolumes in One Canvas
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can pass multiple independent nodes combinations to the ``plot3D`` function while specifying a grid (e.g., ``grid=(2,2)``). This allows you to divide the canvas into multiple independent sub-canvases, where each sub-canvas displays a separate 3D data set within the same canvas. The example code for this can be found in the documentation 
at `cigvis/gallery/3Dvispy/10 <https://cigvis.readthedocs.io/en/latest/gallery/3Dvispy/10-multi_canvas.html#sphx-glr-gallery-3dvispy-10-multi-canvas-py>`_.

.. image:: https://raw.githubusercontent.com/JintaoLee-Roger/images/main/cigvis/3Dvispy/10.gif
   :alt: 10
   :align: center

Furthermore, you can link the cameras of all sub-canvases together (just need pass ``share=True`` to ``plot3D`` function). This means that any rotation, scaling, or slicing performed in one sub-canvas will be mirrored in all other sub-canvases, ensuring that they all exhibit the same changes simultaneously. This feature is highly advantageous when comparing multiple sets of data, such as results from different experiments, results alongside labels, seismic data compared with attributes, and more. 
You can find example code for this functionality in the documentation 
at `cigvis/gallery/3Dvispy/11 <https://cigvis.readthedocs.io/en/latest/gallery/3Dvispy/11-share_cameras.html#sphx-glr-gallery-3dvispy-11-share-cameras-py>`_.

.. image:: https://raw.githubusercontent.com/JintaoLee-Roger/images/main/cigvis/3Dvispy/11.gif
   :alt: 11
   :align: center

These capabilities provide a powerful way to visualize and compare multiple independent 3D data sets within a single canvas using cigvis.



Example Data
---------------

All data used by examples in the 
`gallery <https://cigvis.readthedocs.io/en/latest/gallery/index.html>`_ 
can be download at 
`https://rec.ustc.edu.cn/share/19a16120-5c42-11ee-a0d4-4329aa6b754b <https://rec.ustc.edu.cn/share/19a16120-5c42-11ee-a0d4-4329aa6b754b>`_, 
password: ``1234``.



Example Gallery
------------------

See: `cigvis/gallery <https://cigvis.readthedocs.io/en/latest/gallery/index.html>`_
