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
import vispy
from vispy.util import keys
from vispy.gloo.util import _screenshot
from vispy.visuals import MeshVisual, CompoundVisual
from vispy import scene

import cigvis
from .xyz_axis import XYZAxis
from .axis_aligned_image import AxisAlignedImage
from .colorbar import Colorbar


class VisCanvas(scene.SceneCanvas):
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
        visual_nodes: Union[List, Dict] = [],
        grid: Tuple = None,
        share: bool = False,
        bgcolor: str = 'white',
        cbar_region_ratio: float = 0.125,

        # for camera
        scale_factor: float = None,
        center=None,
        fov: float = 45,
        azimuth: float = 50,
        elevation: float = 50,
        zoom_factor: float = 1.0,
        axis_scales: Tuple = (1.0, 1.0, 1.0),
        auto_range: bool = True,

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
        self.nodes = {}

        self.init_size = size
        self.cbar_ratio = cbar_region_ratio
        self.fov = fov
        self.azimuth = azimuth
        self.elevation = elevation
        self.scale_factor = scale_factor
        self.center = center
        self.auto_range = auto_range
        self.zoom_factor = zoom_factor
        self.share = share

        axis_scales = list(axis_scales)
        for i, r in enumerate(cigvis.is_axis_reversed()):
            axis_scales[i] *= (1 - 2 * r)
        self.axis_scales = axis_scales

        self._init_grid_input(grid, visual_nodes)

        self.add_visual_nodes()

        if self.share:
            self.link_cameras()

        # Attach a ViewBox to a grid and initiate the camera with the given
        # parameters.
        if not (auto_range or (scale_factor and center)):
            raise ValueError("scale_factor and center cannot be None" +
                             " when auto_range=False")

        # Manage the selected visual node.
        self.drag_mode = False
        self.selected = None  # no selection by default
        self.hover_on = None  # visual node that mouse hovers on, None by default
        self.selected2 = []

        if not self.share:
            for view, nodes in zip(self.view, self.nodes.values()):
                self._attach_light(view, nodes)
        else:
            # pass
            # for view in self.view:
            self._attach_light_share(self.view[-1], self.nodes)

        self.freeze()

    def on_mouse_press(self, event):
        # Hold <Alt> and click left to print position
        if keys.ALT in event.modifiers:
            # TODO
            pass

        # Hold <Ctrl> to enter drag mode or press <d> to toggle.
        if keys.CONTROL in event.modifiers or self.drag_mode:
            # Temporarily disable the interactive flag of the ViewBox because it
            # is masking all the visuals. See details at:
            # https://github.com/vispy/vispy/issues/1336
            for view in self.view:
                view.interactive = False
            hover_on = self.visual_at(event.pos)

            if event.button == 1 and self.selected is None:
                # If no previous selection, make a new selection if cilck on a valid
                # visual node, and highlight this node.
                if self._check_drag(hover_on):
                    self.selected = hover_on
                    if self.share:
                        self._get_selected2(self.selected)
                    self.selected.highlight.visible = True
                    # Set the anchor point on this node.
                    self.selected.set_anchor(event)

                # Nothing to do if the cursor is NOT on a valid visual node.

            # Reenable the ViewBox interactive flag.
            for view in self.view:
                view.interactive = True

    def on_mouse_release(self, event):
        # Hold <Ctrl> to enter drag mode or press <d> to toggle.
        if keys.CONTROL in event.modifiers or self.drag_mode:
            if self.selected is not None:
                # Erase the anchor point on this node.
                self.selected.anchor = None
                # Then, deselect any previous selection.
                self.selected = None
                self.selected2 = []

    def on_mouse_move(self, event):
        # Hold <Ctrl> to enter drag mode or press <d> to toggle.
        if keys.CONTROL in event.modifiers or self.drag_mode:
            # Temporarily disable the interactive flag of the ViewBox because it
            # is masking all the visuals. See details at:
            # https://github.com/vispy/vispy/issues/1336
            for view in self.view:
                view.interactive = False
            hover_on = self.visual_at(event.pos)

            if event.button == 1:
                # if self.selected is not None:
                if self._check_drag(self.selected):
                    self.selected.drag_visual_node(event)
                    for node in self.selected2:
                        if isinstance(node, XYZAxis):
                            node._update_axis(self.selected.loc)
                        else:
                            node._update_location(self.selected.pos)
            else:
                # If the left cilck is released, update highlight to the new visual
                # node that mouse hovers on.
                if hover_on != self.hover_on:
                    # de-highlight previous hover_on
                    if self._check_drag(self.hover_on):
                        self.hover_on.highlight.visible = False
                    self.hover_on = hover_on
                    # highlight the new hover_on
                    if self._check_drag(self.hover_on):
                        self.hover_on.highlight.visible = True

            # Reenable the ViewBox interactive flag.
            for view in self.view:
                view.interactive = True

    def on_key_press(self, event):
        # Press <Space> to reset camera.
        if event.text == ' ':
            for view in self.view:
                view.camera.fov = self.fov
                view.camera.azimuth = self.azimuth
                view.camera.elevation = self.elevation
                view.camera.set_range()
                view.camera.scale_factor = self.scale_factor
                view.camera.scale_factor /= self.zoom_factor

                view.camera._flip_factors = self.axis_scales
                view.camera._update_camera_pos()

                for child in view.children:
                    if isinstance(child, XYZAxis):
                        child._update_axis()

        # Press <s> to save a screenshot.
        if event.text == 's':
            screenshot = _screenshot()
            # screenshot = self.render()
            vispy.io.write_png(self.pngDir + self.title + '.png', screenshot)

        # Press <d> to toggle drag mode.
        if event.text == 'd':
            if not self.drag_mode:
                self.drag_mode = True
                for view in self.view:
                    view.camera.viewbox.events.mouse_move.disconnect(
                        view.camera.viewbox_mouse_event)
            else:
                self.drag_mode = False
                self._exit_drag_mode()
                for view in self.view:
                    view.camera.viewbox.events.mouse_move.connect(
                        view.camera.viewbox_mouse_event)

        # Press <a> to get the parameters of all visual nodes.
        if event.text == 'a':
            print("===== All useful parameters ====")
            # Canvas size.
            print("Canvas size = {}".format(self.size))
            # Collect camera parameters.
            print("Camera:")
            camera_state = self.view[0].camera.get_state()
            for key, value in camera_state.items():
                print(" - {} = {}".format(key, value))
            print(" - {} = {}".format('zoom factor', self.zoom_factor))

            # axis scales
            factors = list(self.view[0].camera._flip_factors)
            print(f'axes scale ratio (< 0 means axis reversed):')
            print(f' - x: {factors[0]}')
            print(f' - y: {factors[1]}')
            print(f' - z: {factors[2]}')

            # Collect slice parameters.
            print("Slices:")
            pos_dict = {'x': [], 'y': [], 'z': []}
            for node in self.view[0].scene.children:
                if self._check_drag(node):
                    pos = node.pos
                    pos_dict[node.axis].append(pos)
            for axis, pos in pos_dict.items():
                print(" - {}: {}".format(axis, pos))
            # Collect the axis legend parameters.
            for node in self.view[0].children:
                if isinstance(node, XYZAxis):
                    print("XYZAxis loc = {}".format(node.loc))

        # zoom in z axis, press <z>
        if event.text == 'z':
            for view in self.view:
                factors = list(view.camera._flip_factors)
                factors[2] += (0.2 * (1 - 2 * cigvis.is_z_reversed()))
                view.camera._flip_factors = factors
                view.camera._update_camera_pos()

                self.update()

        # zoom out z axis, press <Z>, i.e. <Shift>+<z>
        if event.text == 'Z':
            for view in self.view:
                factors = list(view.camera._flip_factors)
                factors[2] -= (0.2 * (1 - 2 * cigvis.is_z_reversed()))
                view.camera._flip_factors = factors
                view.camera._update_camera_pos()

            self.update()

        # zoom in fov, press <f>
        if event.text == 'f':
            for view in self.view:
                view.camera.fov += 5

        # zoom out fov, press <F>
        if event.text == 'F':
            for view in self.view:
                view.camera.fov -= 5

    def on_key_release(self, event):
        # Cancel selection and highlight if release <Ctrl>.
        if keys.CONTROL not in event.modifiers:
            self._exit_drag_mode()

    def _exit_drag_mode(self):
        if self._check_drag(self.hover_on):
            self.hover_on.highlight.visible = False
            self.hover_on = None
        if self._check_drag(self.selected):
            self.selected.highlight.visible = False
            self.selected.anchor = None
            self.selected = None
            self.selected2 = []

    # HACK: 直接绑定 ShadingFilter 会不会更方便
    # 使用 mesh._vshare.filters 获取 attach 的 ShadingFilter
    def _attach_light(self, view, nodes):
        light_dir = (0, -1, 0, 0)

        for node in nodes:
            if isinstance(node, MeshVisual):
                if node.shading_filter is not None:
                    node.shading_filter.light_dir = light_dir[:3]
            if isinstance(node, CompoundVisual):
                if hasattr(node, 'meshs'):
                    for mesh in node.meshs:
                        if mesh.shading_filter is not None:
                            mesh.shading_filter.light_dir = light_dir[:3]

        initial_light_dir = view.camera.transform.imap(light_dir)
        view.camera.azimuth = self.azimuth
        view.camera.elevation = self.elevation

        @view.scene.transform.changed.connect
        def on_transform_change(event):
            transform = view.camera.transform
            for node in nodes:
                if isinstance(node, MeshVisual):
                    if node.shading_filter is not None:
                        # print(transform.map(initial_light_dir))
                        node.shading_filter.light_dir = transform.map(
                            initial_light_dir)[:3]
                if isinstance(node, CompoundVisual):
                    if hasattr(node, 'meshs'):
                        for mesh in node.meshs:
                            if mesh.shading_filter is not None:
                                mesh.shading_filter.light_dir = transform.map(
                                    initial_light_dir)[:3]

    def _attach_light_share(self, view, nodess):
        light_dir = (0, -1, 0, 0)

        for nodes in nodess.values():
            for node in nodes:
                if isinstance(node, MeshVisual):
                    if node.shading_filter is not None:
                        node.shading_filter.light_dir = light_dir[:3]
                if isinstance(node, CompoundVisual):
                    if hasattr(node, 'meshs'):
                        for mesh in node.meshs:
                            if mesh.shading_filter is not None:
                                mesh.shading_filter.light_dir = light_dir[:3]
        initial_light_dir = view.camera.transform.imap(light_dir)

        view.camera.azimuth = self.azimuth
        view.camera.elevation = self.elevation

        @view.scene.transform.changed.connect
        def on_transform_change(event):
            transform = view.camera.transform
            for nodes in nodess.values():
                for node in nodes:
                    if isinstance(node, MeshVisual):
                        if node.shading_filter is not None:
                            # print(transform.map(initial_light_dir))
                            node.shading_filter.light_dir = transform.map(
                                initial_light_dir)[:3]
                    if isinstance(node, CompoundVisual):
                        if hasattr(node, 'meshs'):
                            for mesh in node.meshs:
                                if mesh.shading_filter is not None:
                                    mesh.shading_filter.light_dir = transform.map(
                                        initial_light_dir)[:3]

    def _check_drag(self, node):
        """
        Only AxisAlignedImage and XYZAxis can be drag
        """
        return isinstance(node, (AxisAlignedImage, XYZAxis))

    def _get_selected2(self, node):
        """
        uesed when self.share == true
        get the correspanding Nodes of the input node to drag
        """
        assert self._check_drag(node)
        ids = node.ids
        k = node.name[:3]

        for key in self.nodes.keys():
            if key != k:
                self.selected2 += [
                    n for n in self.nodes[key]
                    if self._check_drag(n) and n.ids == ids
                ]

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

            if isinstance(node, XYZAxis):
                # Set the parent to view, instead of view.scene,
                # so that this legend will stay at its location on
                # the canvas, and won't rotate away.
                node.parent = view
                node.canvas_size = self.size
                self.events.resize.connect(node.on_resize)
                node.highlight.parent = view
                node._update_axis()
                self.events.mouse_move.connect(node.on_mouse_move)

        if self.auto_range:
            view.camera.set_range()

        if self.scale_factor is None:
            self.scale_factor = view.camera.scale_factor

        view.camera.scale_factor /= self.zoom_factor

        return view

    def _add_colorbar(self, view, colorbar):
        """
        Create a secondary ViewBox to host the Colorbar visual.
        Make it solid background, image from primary ViewBox shall be
        blocked if overlapping.
        """
        view.interactive = True  # disable so that it won't be selectable

        # use PanZoomCamera so we can draw a **high qaulity** colorbar image
        view.camera = scene.cameras.PanZoomCamera(aspect=1)
        # disable mouse drag, keep zoom in/out
        view.camera.viewbox.events.mouse_move.disconnect(
            view.camera.viewbox_mouse_event)
        colorbar.parent = view.scene
        view.camera.flip = (0, 1, 0)
        view.camera.set_range()

    def link_cameras(self):
        """
        link all cameras
        """

        for view in self.view[:-1]:
            view.camera.link(self.view[-1].camera)
