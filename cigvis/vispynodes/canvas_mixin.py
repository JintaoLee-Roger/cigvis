# Copyright (c) 2024 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.

from numbers import Integral, Real

from vispy.util import keys

import cigvis
from .indicator import XYZAxis, NorthPointer
from .axis_aligned_image import AxisAlignedImage
from .screenshot import _save_canvas_png


def _round_float(value, ndigits=6):
    value = float(value)
    if abs(value) < 10 ** (-(ndigits + 1)):
        value = 0.0
    return round(value, ndigits)


def _python_value(value, ndigits=6):
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, tuple):
        return tuple(_python_value(v, ndigits) for v in value)
    if isinstance(value, list):
        return [_python_value(v, ndigits) for v in value]
    if isinstance(value, dict):
        return {k: _python_value(v, ndigits) for k, v in value.items()}
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return _round_float(value, ndigits)
    return value


def _literal(value):
    return repr(_python_value(value))


def _print_kw(name, value, indent='    '):
    print(f'{indent}{name}={_literal(value)},')


def _api_axis_scales(factors):
    signs = [1 - 2 * reversed_ for reversed_ in cigvis.is_axis_reversed()]
    return tuple(factor * sign for factor, sign in zip(factors, signs))


class EventMixin:

    def _scene_views(self):
        return getattr(self, 'view', None) or []

    def _get_xyz_from_event(self, event):
        hover_on = self.visual_at(event.pos)
        if hasattr(hover_on, 'get_click_pos3d'):
            xyz = hover_on.get_click_pos3d(event)
            return xyz, hover_on
        return None, hover_on

    def on_mouse_press(self, event):
        views = self._scene_views()
        if not views:
            self.drag_mode = False
            return

        # Hold <Alt> and click left to print position
        if (event.button == 1) and (keys.ALT in event.modifiers) and (keys.CONTROL not in event.modifiers) and (not self.drag_mode):
            hover_on = self.visual_at(event.pos)
            if hasattr(hover_on, 'get_click_pos3d'):
                xyz = hover_on.get_click_pos3d(event)

                if self._prompt_callback is None:
                    print(hover_on.axis, xyz)
                else:
                    self._prompt_callback(xyz, hover_on, event)

        # Hold <Ctrl> to enter drag mode or press <d> to toggle.
        if keys.CONTROL in event.modifiers or self.drag_mode:
            # Temporarily disable the interactive flag of the ViewBox because it
            # is masking all the visuals. See details at:
            # https://github.com/vispy/vispy/issues/1336
            for view in views:
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
            for view in views:
                view.interactive = True

    def on_mouse_release(self, event):
        if not self._scene_views():
            self.drag_mode = False
            return

        # Hold <Ctrl> to enter drag mode or press <d> to toggle.
        if keys.CONTROL in event.modifiers or self.drag_mode:
            if self.selected is not None:
                # Erase the anchor point on this node.
                self.selected.anchor = None
                # Then, deselect any previous selection.
                self.selected = None
                self.selected2 = []

    def on_mouse_move(self, event):
        views = self._scene_views()
        if not views:
            self.drag_mode = False
            return

        # Hold <Ctrl> to enter drag mode or press <d> to toggle.
        if keys.CONTROL in event.modifiers or self.drag_mode:
            # Temporarily disable the interactive flag of the ViewBox because it
            # is masking all the visuals. See details at:
            # https://github.com/vispy/vispy/issues/1336
            for view in views:
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
            for view in views:
                view.interactive = True

    def on_key_press(self, event):
        if not hasattr(self, 'keymove'):
            self.unfreeze()
            self.keymove = 0
            self.freeze()
        views = self._scene_views()
        # Press <Space> to reset camera.
        if event.text == ' ':
            if not views:
                return
            for view in views:
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
            _save_canvas_png(
                self,
                self.title + '.png',
                self.pngDir,
                getattr(self, '_shortcut_save_kw', {
                    'transparent_bg': True,
                }),
            )

        # Press <d> to toggle drag mode.
        if event.text == 'd':
            if not views:
                return
            if not self.drag_mode:
                self.drag_mode = True
                for view in views:
                    view.camera.viewbox.events.mouse_move.disconnect(
                        view.camera.viewbox_mouse_event)
            else:
                self.drag_mode = False
                self._exit_drag_mode()
                for view in views:
                    view.camera.viewbox.events.mouse_move.connect(
                        view.camera.viewbox_mouse_event)

        # Press <a> to get the parameters of all visual nodes.
        if event.text == 'a':
            if not views:
                return
            print("===== Copyable cigvis state =====")
            camera_state = views[0].camera.get_state()
            factors = list(views[0].camera._flip_factors)
            view_kwargs = {
                'size': tuple(self.size),
                'scale_factor': camera_state.get('scale_factor'),
                'center': camera_state.get('center'),
                'fov': camera_state.get('fov'),
                'azimuth': camera_state.get('azimuth'),
                'elevation': camera_state.get('elevation'),
                'axis_scales': _api_axis_scales(factors),
            }
            if getattr(self, 'nrows', 1) > 1 or getattr(self, 'ncols', 1) > 1:
                view_kwargs['grid'] = (self.nrows, self.ncols)
                view_kwargs['share'] = self.share

            print("# Paste into cigvis.plot3D(...):")
            print("view=cigvis.Plot3DView(")
            for key, value in view_kwargs.items():
                if value is not None:
                    _print_kw(key, value)
            print("),")

            pos_dict = {'x': [], 'y': [], 'z': []}
            for node in views[0].scene.children:
                axis = getattr(node, 'axis', None)
                if self._check_drag(node) and axis in pos_dict:
                    pos_dict[axis].append(_python_value(node.pos))

            print("")
            print("# Paste into cigvis.create_slices(...):")
            _print_kw('pos', pos_dict)

            xyz_axis_locs = []
            for node in views[0].children:
                if isinstance(node, XYZAxis):
                    xyz_axis_locs.append(_python_value(node.loc))
            if xyz_axis_locs:
                print("")
                print("# Current axis legend location(s):")
                _print_kw('xyz_axis_loc', xyz_axis_locs)

            print("")
            print("# Raw camera state:")
            for key, value in camera_state.items():
                _print_kw(key, value)
            _print_kw('zoom_factor', self.zoom_factor)
            _print_kw('camera_axis_scales', tuple(factors))

        # zoom in z axis, press <z>
        if event.text == 'z':
            if not views:
                return
            for view in views:
                factors = list(view.camera._flip_factors)
                factors[2] += (0.2 * (1 - 2 * cigvis.is_z_reversed()))
                view.camera._flip_factors = factors
                view.camera._update_camera_pos()

                self.update()

        # zoom out z axis, press <Z>, i.e. <Shift>+<z>
        if event.text == 'Z':
            if not views:
                return
            for view in views:
                factors = list(view.camera._flip_factors)
                factors[2] -= (0.2 * (1 - 2 * cigvis.is_z_reversed()))
                view.camera._flip_factors = factors
                view.camera._update_camera_pos()

            self.update()

        # zoom in fov, press <f>
        if event.text == 'f':
            if not views:
                return
            for view in views:
                view.camera.fov += 5

        # zoom out fov, press <F>
        if event.text == 'F':
            if not views:
                return
            for view in views:
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
