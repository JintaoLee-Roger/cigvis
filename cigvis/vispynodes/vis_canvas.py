# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2023, modified by Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC)
#
#
# Copyright (C) 2019 Yunzhi Shi @ The University of Texas at Austin.
# All rights reserved.
# Distributed under the MIT License. See LICENSE for more info.
# -----------------------------------------------------------------------------

from typing import Dict, List, Tuple, Union
import warnings
from vispy import scene

import cigvis
from .indicator import XYZAxis, NorthPointer
from .axis3d import Axis3D
from .colorbar import Colorbar
from .canvas_mixin import EventMixin, LightMixin, AxisMixin


class VisCanvas(scene.SceneCanvas, EventMixin, LightMixin, AxisMixin):
    """
    A canvas that automatically draw all contents in a 3D seismic
    visualization scene, which may include 3D seismic volume slices, axis
    legend, colorbar, etc.

    Parameters
    ----------
    size : Tuple
        canvas size
    visual_nodes : Union[List, Dict]
        nodes, can be a List like: [mesh1, mesh2, ...] (for a single widget), 
        or [[mesh1, mesh2, ...], [image1, mesh3, ...], ...] 
        (must input 'grid'). It also can be a Dict like: 
        {'0,0': [mesh1, img1, ...], '0,1': [mesh2, ...]}. Its keys represent
        the location in the grid
    grid : Tuple
        a 2D Tuple, (nrows, ncols)
    share : bool
        whether link all cameras when 'grid' is not None
    bgcolor : str or Color
        background color
    cbar_region_ratio : float
        colorbar region ration, i.e., (width*cbar_region_ratio, hight)
    scale_factor : float
        camera scale factor
    center : Tuple
        center position, default is None
    fov : float
        camera fov
    azimuth : float
        camera azimuth
    elevation : float
        camera elevation
    zoom_factor : float
        camera zoom factor
    axis_scales : Tuple
        axis scale, default is (1, 1, 1)
    auto_range : bool
        default is True
    
    savedir : str
        the dir to save sreenshot when press <s>
    title : str
        canvas title name, which is also used as the save screenshot name 
    """

    def __init__(
        self,
        size: Tuple = (800, 720),
        visual_nodes: Union[List, Dict] = None,
        grid: Tuple = None,
        share: bool = False,
        bgcolor: str = 'white',
        cbar_region_ratio: float = 0.125,

        # for camera
        scale_factor: float = None,
        center=None,
        fov: float = 30,
        azimuth: float = 50,
        elevation: float = 50,
        zoom_factor: float = 1.0,
        axis_scales: Tuple = (1.0, 1.0, 1.0),
        auto_range: bool = True,

        # for light
        dyn_light: bool = True,

        # for save
        savedir: str = './',
        title: str = 'Seismic3D',
    ):

        self.pngDir = savedir

        # Create a SceneCanvas obj and unfreeze it so we can add more
        # attributes inside.
        scene.SceneCanvas.__init__(self,
                                   title=title,
                                   keys='interactive',
                                   size=size,
                                   bgcolor=bgcolor)

        self.unfreeze()
        if not auto_range:
            warnings.warn("`auto_range` is deprecated and will be remove in the future version. Just ignore this parameter.", DeprecationWarning, stacklevel=2) # yapf: disable

        self.nodes = {}

        self.init_size = size
        self.cbar_ratio = cbar_region_ratio
        self.fov = fov
        self.azimuth = azimuth
        self.elevation = elevation
        self.scale_factor = scale_factor
        self.center = center
        self.zoom_factor = zoom_factor
        self.share = share

        self.dyn_light = dyn_light

        axis_scales = list(axis_scales)
        for i, r in enumerate(cigvis.is_axis_reversed()):
            axis_scales[i] *= (1 - 2 * r)
        self.axis_scales = axis_scales

        # Manage the selected visual node.
        self.drag_mode = False
        self.selected = None  # no selection by default
        self.hover_on = None  # visual node that mouse hovers on, None by default
        self.selected2 = []

        self.freeze()

        if visual_nodes is not None:
            self.add_nodes(visual_nodes, grid)

    def update_camera(self, azimuth, elevation, fov):
        self.azimuth = azimuth
        self.elevation = elevation
        self.fov = fov

        if not hasattr(self, 'view'):
            return

        for view in self.view:
            view.camera.azimuth = self.azimuth
            view.camera.elevation = self.elevation
            view.camera.fov = self.fov

    def update_axis_scales(self, axis_scales):
        axis_scales = list(axis_scales)
        for i, r in enumerate(cigvis.is_axis_reversed()):
            axis_scales[i] *= (1 - 2 * r)
        self.axis_scales = axis_scales

        if not hasattr(self, 'view'):
            return

        for view in self.view:
            view.camera._flip_factors = self.axis_scales
            view.camera._update_transform()
            self.update()

    def add_nodes(self, visual_nodes: Union[List, Dict], grid: Tuple = None):

        self.unfreeze()

        self._init_grid_input(grid, visual_nodes)

        self.add_visual_nodes()

        if self.share:
            self.link_cameras()

        if not self.share:
            for view, nodes in zip(self.view, self.nodes.values()):
                self._attach_light(view, nodes)
        else:
            self._attach_light_share(self.view[-1], self.nodes)

        self.freeze()

    def _init_grid_input(self, grid, visual_nodes):
        """
        inital grid (get self.nrows and self.ncols),
        and convert visual_nodes to self.nodes (Dict)

        self.nodes is a dict like {'i,j': nodes}, its
        key 'i,j' represents a loctaion (i, j) in the grid
        """
        if grid is not None:
            self.nrows = grid[0]
            self.ncols = grid[1]
            assert len(visual_nodes) <= grid[0] * grid[1]
            if isinstance(visual_nodes, List):
                for i, nodes in enumerate(visual_nodes):
                    assert isinstance(nodes, List)
                    r = i // grid[1]
                    c = i - r * grid[1]
                    key = f'{r},{c}'
                    self.nodes[key] = nodes
            elif isinstance(visual_nodes, Dict):
                ks = list(visual_nodes.keys())
                for k in ks:
                    r, c = [int(i) for i in k.split(',')]
                    if not (r >= 0 and r < grid[0] and c >= 0 and c < grid[1]):
                        raise KeyError(
                            f'visual_nodes key: {ks} is out bound of grid: {grid}'
                        )
                self.nodes = visual_nodes
            else:
                raise ValueError('Invalid type of visual_nodes')
        else:
            self.nrows = 1
            self.ncols = 1
            if isinstance(visual_nodes[0], List):
                self.nodes['0,0'] = visual_nodes[0]
            elif isinstance(visual_nodes, Dict):
                self.nodes = visual_nodes
            else:
                self.nodes['0,0'] = visual_nodes

    def add_visual_nodes(self):
        """
        Add all visual nodes
        """
        # Create a Grid widget on the canvas to host separate Viewbox (e.g.,
        # 3D image on the left panel and colorbar to the right).
        grid = self.central_widget.add_grid()
        N = len(self.nodes)
        w, h = self.init_size
        self.view = []

        for i in range(self.nrows):
            for j in range(self.ncols):
                k = f'{i},{j}'
                if k not in self.nodes.keys():
                    continue

                widget = grid.add_widget(row=i,
                                         col=j,
                                         size=(w / self.ncols, h / self.nrows))
                cgrid = widget.add_grid()
                view = cgrid.add_view(row=0, col=0)
                view = self._add_nodes(view, self.nodes[k], k)
                self.view.append(view)

                # TODO: change view2.width_max when resize
                for node in self.nodes[k]:
                    if isinstance(node, Colorbar):
                        view2 = cgrid.add_view(row=0, col=1)
                        view2.width_max = self.cbar_ratio * cgrid.width
                        self._add_colorbar(view2, node)

    def _add_nodes(self, view, nodes, k: str):
        """
        add nodes to a viewbox
        """
        # set azimuth=0 and elevation=0, for convenient lighting
        # change them in self._attach_light() function
        view.camera = scene.cameras.TurntableCamera(
            # self.camera = scene.cameras.ArcballCamera(
            scale_factor=self.scale_factor,
            center=self.center,
            fov=self.fov,
            azimuth=0,
            elevation=0,
        )

        view.camera._flip_factors = self.axis_scales.copy()

        # Attach all main visual nodes (e.g. slices, meshs, volumes) to the ViewBox.
        for i, node in enumerate(nodes):
            if not isinstance(node, Colorbar):
                node.name = k + f'-{i}'
                view.add(node)

            if isinstance(node, (XYZAxis, NorthPointer)):
                # Set the parent to view, instead of view.scene,
                # so that this legend will stay at its location on
                # the canvas, and won't rotate away.
                node.parent = view
                node.canvas_size = self.size
                self.events.resize.connect(node.on_resize)
                if hasattr(node, 'highlight'):
                    node.highlight.parent = view
                if isinstance(node, NorthPointer):
                    node.loc = (80, self.size[1] - 80)
                node._update_axis()
                self.events.mouse_move.connect(node.on_mouse_move)

            if isinstance(node, Axis3D) and node.auto_change:
                self._change_pos(view, node)

        view.camera.set_range()
        if self.center is not None:
            view.camera.center = self.center
        if self.scale_factor is not None:
            view.camera.scale_factor = self.scale_factor

        if self.scale_factor is None:
            self.scale_factor = view.camera.scale_factor
        if self.center is None:
            self.center = view.camera.center

        view.camera.scale_factor /= self.zoom_factor

        return view

    def _add_colorbar(self, view, colorbar):
        """
        Create a secondary ViewBox to host the Colorbar visual.
        Make it solid background, image from primary ViewBox shall be
        blocked if overlapping.
        """
        view.interactive = False  # disable so that it won't be selectable
        colorbar.parent = view.scene

        # use PanZoomCamera so we can draw a **high qaulity** colorbar image
        # view.camera = scene.cameras.PanZoomCamera(aspect=1)

        # disable mouse drag, keep zoom in/out
        # view.camera.viewbox.events.mouse_move.disconnect(
        #     view.camera.viewbox_mouse_event)

        colorbar.pos = (0, self.size[1] / 2 / self.nrows)
        colorbar.canvas_size = self.size
        # view.camera.flip = (0, 1, 0)
        # view.camera.set_range()
        self.events.resize.connect(colorbar.on_resize)

    def link_cameras(self):
        """
        link all cameras
        """
        for view in self.view[:-1]:
            view.camera.link(self.view[-1].camera)

    def add_node(self, node):
        """
        Add a node to the canvas.
        NOTE: this function is valid only when one canvas, i.e., self.nrows and self.ncols are both 1
        """
        if self.nrows != 1 or self.ncols != 1:
            raise ValueError(
                "This function is valid only when one canvas, i.e., self.nrows and self.ncols are both 1"
            )

        self.unfreeze()
        node.name = '0-0'
        self.nodes['0,0'].append(node)
        self.view[0].add(node)
        if hasattr(node, 'shading_filter'):
            node.shading_filter.light_dir = self.view[0].camera.transform.map(self.initial_light_dir)[:3] # yapf: disable
        self.freeze()

    def remove_node(self, node, delete=True):
        """
        Remove a node from the canvas.
        NOTE: this function is valid only when one canvas, i.e., self.nrows and self.ncols are both 1
        """
        if self.nrows != 1 or self.ncols != 1:
            raise ValueError(
                "This function is valid only when one canvas, i.e., self.nrows and self.ncols are both 1"
            )

        node.parent = None
        for n in reversed(self.nodes['0,0']):
            if n == node:
                self.nodes['0,0'].remove(n)
                break
        if delete:
            del node