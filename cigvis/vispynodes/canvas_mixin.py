# Copyright (c) 2024 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.

import vispy
from vispy.util import keys
from vispy.gloo.util import _screenshot

import cigvis
from .indicator import XYZAxis, NorthPointer
from .axis_aligned_image import AxisAlignedImage
from vispy.visuals import MeshVisual, CompoundVisual


class EventMixin:

    def on_mouse_press(self, event):
        # Hold <Alt> and click left to print position
        if keys.ALT in event.modifiers:
            ## 屏幕/画布坐标系 Canvas Coordinates
            # print(event.pos)

            hover_on = self.visual_at(event.pos)
            if hasattr(hover_on, 'get_click_pos3d'):
                print(hover_on.get_click_pos3d(event))

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
        if not hasattr(self, 'keymove'):
            self.unfreeze()
            self.keymove = 0
            self.freeze()
        # Press <Space> to reset camera.
        if event.text == ' ':
            for view in self.view:
                view.camera.fov = self.fov
                view.camera.azimuth = self.azimuth
                view.camera.elevation = self.elevation
                view.camera.set_range()
                view.camera.center = self.center
                view.camera.scale_factor = self.scale_factor
                view.camera.scale_factor /= self.zoom_factor

                view.camera._flip_factors = self.axis_scales
                view.camera._update_camera_pos()

                for child in view.children:
                    if isinstance(child, (XYZAxis, NorthPointer)):
                        child._update_axis()

        # Press <s> to save a screenshot.
        if event.text == 's':
            # viewport = list(gl.glGetParameter(gl.GL_VIEWPORT))
            # border = (viewport[3] - self.size[1]) // 2
            # viewport[0] = viewport[2] - self.size[0] - border
            # viewport[1] = viewport[3] - self.size[1] - border
            # viewport[2] = self.size[0]
            # viewport[3] = self.size[1]
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

        if event.key == keys.LEFT:
            self.keymove = (self.keymove - 1) % 3
        if event.key == keys.RIGHT:
            self.keymove = (self.keymove + 1) % 3
        if event.key == keys.UP:
            for nodes in self.nodes.values():
                for node in nodes:
                    if isinstance(node, AxisAlignedImage):
                        if node.axis == ['x', 'y', 'z'][self.keymove]:
                            node._update_location(
                                node.pos + 10)  # TODO: control the step size?
        if event.key == keys.DOWN:
            for nodes in self.nodes.values():
                for node in nodes:
                    if isinstance(node, AxisAlignedImage):
                        if node.axis == ['x', 'y', 'z'][self.keymove]:
                            node._update_location(node.pos - 10)

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


class LightMixin:
    # HACK: 直接绑定 ShadingFilter 会不会更方便
    # 使用 mesh._vshare.filters 获取 attach 的 ShadingFilter
    def _attach_light(self, view, nodes):
        light_dir = (0, -1, 0, 0)

        initial_light_dir = view.camera.transform.imap(light_dir)
        if not hasattr(self, "initial_light_dir"):
            self.initial_light_dir = initial_light_dir
        view.camera.azimuth = self.azimuth
        view.camera.elevation = self.elevation

        for node in nodes:
            if isinstance(node, MeshVisual):
                if node.shading_filter is not None:
                    node.shading_filter.light_dir = view.camera.transform.map(
                        initial_light_dir)[:3]
            if isinstance(node, CompoundVisual):
                if hasattr(node, 'meshs'):
                    for mesh in node.meshs:
                        if mesh.shading_filter is not None:
                            mesh.shading_filter.light_dir = view.camera.transform.map(
                                initial_light_dir)[:3]

        @view.scene.transform.changed.connect
        def on_transform_change(event):
            if not self.dyn_light:
                return
            transform = view.camera.transform
            for node in nodes:
                if hasattr(node, 'dyn_light') and not node.dyn_light:
                    continue
                if isinstance(node, MeshVisual):
                    if node.shading_filter is not None:
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
        initial_light_dir = view.camera.transform.imap(light_dir)
        if not hasattr(self, "initial_light_dir"):
            self.initial_light_dir = initial_light_dir
        view.camera.azimuth = self.azimuth
        view.camera.elevation = self.elevation

        for nodes in nodess.values():
            for node in nodes:
                if isinstance(node, MeshVisual):
                    if node.shading_filter is not None:
                        node.shading_filter.light_dir = view.camera.transform.map(
                            initial_light_dir)[:3]
                if isinstance(node, CompoundVisual):
                    if hasattr(node, 'meshs'):
                        for mesh in node.meshs:
                            if mesh.shading_filter is not None:
                                mesh.shading_filter.light_dir = view.camera.transform.map(
                                    initial_light_dir)[:3]

        @view.scene.transform.changed.connect
        def on_transform_change(event):
            if not self.dyn_light:
                return
            transform = view.camera.transform
            for nodes in nodess.values():
                for node in nodes:
                    if hasattr(node, 'dyn_light') and not node.dyn_light:
                        continue
                    if isinstance(node, MeshVisual):
                        if node.shading_filter is not None:
                            node.shading_filter.light_dir = transform.map(
                                initial_light_dir)[:3]
                    if isinstance(node, CompoundVisual):
                        if hasattr(node, 'meshs'):
                            for mesh in node.meshs:
                                if mesh.shading_filter is not None:
                                    mesh.shading_filter.light_dir = transform.map(
                                        initial_light_dir)[:3]


class AxisMixin:

    def _change_pos(self, view, node):

        @view.scene.transform.changed.connect
        def on_transform_change(event):
            if view.camera.azimuth < -6 and view.camera.azimuth >= -90 - 6:
                if view.camera.elevation > -10:
                    node.update_ticks_pos([3, 1, 0])
                else:
                    node.update_ticks_pos([1, 3, 0])
            elif view.camera.azimuth >= -6 and view.camera.azimuth <= 90 - 6:
                if view.camera.elevation > -10:
                    node.update_ticks_pos([3, 3, 1])
                else:
                    node.update_ticks_pos([1, 1, 1])
            elif view.camera.azimuth > 90 - 6 and view.camera.azimuth < 180 - 6:
                if view.camera.elevation > -10:
                    node.update_ticks_pos([1, 3, 3])
                else:
                    node.update_ticks_pos([3, 1, 3])
            elif view.camera.azimuth > -180 and view.camera.azimuth < -90 - 6:
                if view.camera.elevation > -10:
                    node.update_ticks_pos([1, 1, 2])
                else:
                    node.update_ticks_pos([3, 3, 2])

            node.update_axis()
