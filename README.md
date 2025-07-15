![im1](https://raw.githubusercontent.com/JintaoLee-Roger/images/main/cigvis/more_demos/070.png)

# CIGVis - a tool for visualizing multidimensional geophysical data

## Overview

**cigvis** is a tool for visualizing multidimensional geophysical data, developed by the [Computational Interpretation Group (CIG)](https://cig.ustc.edu.cn/main.htm). Users can quickly visualize data with just a few lines of code.

cigvis can be used for various geophysical data visualizations, including 3D seismic data, overlays of seismic data with other information like labels, faults, RGT, horizon surfaces, well log trajectories, and well log curves, 3D geological bodies, 2D data, and 1D data, among others. Its GitHub repository can be found at [github.com/JintaoLee-Roger/cigvis](https://github.com/JintaoLee-Roger/cigvis), and documentation is available at [https://cigvis.readthedocs.io/](https://cigvis.readthedocs.io/).

cigvis leverages the power of underlying libraries such as [vispy](https://github.com/vispy/vispy) for 3D visualization, [matplotlib](https://matplotlib.org/) for 2D and 1D visualization, [plotly](https://plotly.com/) for Jupyter environments (work in progress), and [viser](https://github.com/nerfstudio-project/viser) for web-based visualization (SSH-Friendly). The 3D visualization component is heavily based on the code from [yunzhishi/seismic-canvas](https://github.com/yunzhishi/seismic-canvas) and has been further developed upon this foundation.

**CIGVis: An open-source Python tool for the real-time interactive visualization of multidimensional geophysical data**    
Jintao Li, Yunzhi Shi, Xinming Wu    
Paper: [https://library.seg.org/doi/abs/10.1190/geo2024-0041.1](https://library.seg.org/doi/abs/10.1190/geo2024-0041.1)    


## Installation

To install via PyPI, use:
```shell
# only with vispy and PyQt5, for desktop
pip install cigvis

# only with plotly, for jupyter
pip install "cigvis[plotly]"

# only with viser, visualizing in browser
pip install "cigvis[viser]"

# install all dependencies
pip install "cigvis[all]"
```

For local installation, clone the repository from GitHub and then install it using pip:
```shell
git clone https://github.com/JintaoLee-Roger/cigvis.git
# only with vispy and PyQt5, for desktop
pip install -e . --config-settings editable_mode=compat

# only with plotly, for jupyter 
pip install -e ".[plotly]" --config-settings editable_mode=compat

# only with viser, visualizing in browser
pip install -e ".[viser]" --config-settings editable_mode=compat

# install all dependencies
pip install -e ".[all]" --config-settings editable_mode=compat
```

## Core Features

1. Convenient 3D geophysical data visualization.
2. Ongoing development of 2D and 1D data visualization.
3. Additional colormaps tailored for geophysical data.
4. Rapid display of large data in OpenVDS format.


## Usage

### Basic Structure

The fundamental structure of cigvis's visualization code consists of:
1. Data loading
2. Creating nodes
3. Passing nodes to the `plot3D` function

For example:
```python
import numpy as np
import cigvis

# Load data
d = np.fromfile('sx.dat', np.float32).reshape(ni, nt, nx)

# Create nodes
nodes = cigvis.create_slices(d)

# Visualize in 3D
cigvis.plot3D(nodes)
```

This basic code structure allows you to quickly visualize your geophysical data using cigvis. Simply load your data, create nodes, and pass them to the `plot3D` function as demonstrated in the example above.

### Camera and Dragging

Left click and drag to rotate the camera angle; right click and drag, or scroll mouse wheel, to zoom in and out. Hold `<Shift>` key, left click and drag to pan move. Press `<Space>` key to return to the initial view. Press `<S>` key to save a screenshot PNG file at any time. Press `<Esc>` key to close the window.

Hold `<Ctrl>` key, the selectable visual nodes will be highlighted when your mouse hovers over them; left click and drag to move the highlighted visual node. The volume slices will update their contents in real-time during dragging. You can also press `<D>` key to toggle the dragging mode on/off.

Press `<z>` to zoom in z axis, press `<Z>` or `<Shift> + <z>` to zoom out z axis. 
Press `<f>` to increase `fov` value, press `<F>` or `<Shift> + <f>` to decrease `fov` value.

![ex1](https://raw.githubusercontent.com/JintaoLee-Roger/images/main/cigvis/ex.gif)

Press `<s>` to save a screen shot.

Press `<a>` to print the camera's parameters in real-time; hold on 
the `<alt>` (or `<option>` in macos) and left click the mouse to 
show the coordinate of the click point in the 3D volume.

![ex2](https://raw.githubusercontent.com/JintaoLee-Roger/images/main/cigvis/ex2.gif)


### Various Geophysical data

In cigvis, we represent various geophysical data as individual nodes, assemble these nodes into a list, and then pass this list to the `plot3D` function for visualization.

We visualize a three-dimensional data volume as multiple slices along the x, y, and z directions. Additionally, we can overlay other three-dimensional data slices on these slices, allowing users to interactively drag them along an axis using the mouse.

Horizon data can be represented as scatter points with a shape of (N, 3), or as z-values on a regular grid of size (n1, n2).

Well log trajectories are displayed as tubes, where the size of the first well log curve is represented by the color and radius at each position along the tube. Other well log curves are displayed as surfaces attached to the tube's edge. An example is shown below (code available at [cigvis/gallery/3Dvispy/09](https://cigvis.readthedocs.io/en/latest/gallery/3Dvispy/09-slice_surf_body_logs.html#sphx-glr-gallery-3dvispy-09-slice-surf-body-logs-py)).

![09](https://raw.githubusercontent.com/JintaoLee-Roger/images/main/cigvis/3Dvispy/09.png)

These capabilities within cigvis allow for versatile and interactive visualizations of a wide range of geophysical data types, enhancing the understanding and analysis of such data in geoscience applications.


### Multivolumes in One Canvas

You can pass multiple independent nodes combinations to the `plot3D` function while specifying a grid (e.g., `grid=(2,2)`). This allows you to divide the canvas into multiple independent sub-canvases, where each sub-canvas displays a separate 3D data set within the same canvas. The example code for this can be found in the documentation at [cigvis/gallery/3Dvispy/10](https://cigvis.readthedocs.io/en/latest/gallery/3Dvispy/10-multi_canvas.html#sphx-glr-gallery-3dvispy-10-multi-canvas-py).

![10](https://raw.githubusercontent.com/JintaoLee-Roger/images/main/cigvis/3Dvispy/10.gif)

Furthermore, you can link the cameras of all sub-canvases together (just need pass `share=True` to `plot3D` function). This means that any rotation, scaling, or slicing performed in one sub-canvas will be mirrored in all other sub-canvases, ensuring that they all exhibit the same changes simultaneously. This feature is highly advantageous when comparing multiple sets of data, such as results from different experiments, results alongside labels, seismic data compared with attributes, and more. 
You can find example code for this functionality in the documentation at [cigvis/gallery/3Dvispy/11](https://cigvis.readthedocs.io/en/latest/gallery/3Dvispy/11-share_cameras.html#sphx-glr-gallery-3dvispy-11-share-cameras-py).

![11](https://raw.githubusercontent.com/JintaoLee-Roger/images/main/cigvis/3Dvispy/11.gif)

These capabilities provide a powerful way to visualize and compare multiple independent 3D data sets within a single canvas using cigvis.

## Web-based Visualization

Based on [viser](https://github.com/nerfstudio-project/viser), cigvis also supports visualization 3D data in web/browser environment with just a few lines changed. All you need to do is simply replace `cigvis` with `viserplot`, see follows:

``` diff
    import numpy as np
    import cigvis
+   from cigvis import viserplot

    # Load data
    d = np.fromfile('sx.dat', np.float32).reshape(ni, nt, nx)

    # Create nodes
-   nodes = cigvis.create_slices(d)
+   nodes = viserplot.create_slices(d)

    # Visualize in 3D
-   cigvis.plot3D(nodes)
+   viserplot.plot3D(nodes)
```

When you are in `jupyter` environment, we recommand to maintain a unique server, otherwise the port will be changed.

``` diff
    import numpy as np
    import cigvis
+   from cigvis import viserplot
+   server = viserplot.create_server(8080)

    # Load data
    d = np.fromfile('sx.dat', np.float32).reshape(ni, nt, nx)

    # Create nodes
-   nodes = cigvis.create_slices(d)
+   nodes = viserplot.create_slices(d)

    # Visualize in 3D
-   cigvis.plot3D(nodes)
+   viserplot.plot3D(nodes, server=server)
```

After calling `viserplot.plot3D`, it will generate some logs like:
```txt
╭─────────────── viser ───────────────╮
│             ╷                       │
│   HTTP      │ http://0.0.0.0:8080   │
│   Websocket │ ws://0.0.0.0:8080     │
│             ╵                       │
╰─────────────────────────────────────╯
```

If you are running the code on your local machine, just open `0.0.0.0:8080` in your browser to see the image. If you are running the code on a remote server (yes, cigvis also works when connected remotely using ssh!), you can enter `{ip}:8080` in the browser to see the visualization (`ip` is the ip of remote server, e.g., `222.195.77.88:8080`).

It is not possible to divide multiple canvases in the browser, i.e. it is not possible to visualize multiple canvases in a tab, which may cause some trouble when comparing different results. But we can use some tricks to achieve this, see this example ([cigvis/gallery/viser/04](https://cigvis.readthedocs.io/en/latest/gallery/viser/04_comparison.html#sphx-glr-gallery-viser-04-comparison-py)):

![04](https://raw.githubusercontent.com/JintaoLee-Roger/images/main/cigvis/viser/04.gif)

There are sevreal examples in [gallery/viser](https://cigvis.readthedocs.io/en/latest/gallery/viser/index.html) for reference.



## Citations

If you find this work useful in your research and want to cite it, please consider use this:

Plain Text
```text
Li, J., Shi, Y. and Wu, X., 2024. CIGVis: an open-source python tool for real-time interactive visualization of multidimensional geophysical data. Geophysics, 90(1), pp.1-37.
```

BibTex
```
@article{li2024cigvis,
  title={CIGVis: an open-source python tool for real-time interactive visualization of multidimensional geophysical data},
  author={Li, Jintao and Shi, Yunzhi and Wu, Xinming},
  journal={Geophysics},
  volume={90},
  number={1},
  pages={1--37},
  year={2024},
  publisher={Society of Exploration Geophysicists}
}
```



## Example Data

All data used by examples in the [gallery](https://cigvis.readthedocs.io/gallery) can be download at [https://rec.ustc.edu.cn/share/19a16120-5c42-11ee-a0d4-4329aa6b754b](https://rec.ustc.edu.cn/share/19a16120-5c42-11ee-a0d4-4329aa6b754b), password: `1234`.


## Example Gallery

See: [cigvis/gallery](https://cigvis.readthedocs.io/en/latest/gallery/index.html)


