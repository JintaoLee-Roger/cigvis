# Copyright (c) 2025 Jintao Li.
# University of Science and Technology of China (USTC).
# All rights reserved.

from numbers import Integral, Real
from typing import List
import viser
import numpy as np
from .volume_slice import VolumeSlice
from .meshnode import MeshNode
from .well_log import LogBase
from cigvis import colormap
from packaging import version
import imageio.v3 as iio
from PIL import Image, ImageDraw
import re


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


def _infer_init_scale(nodes):
    for node in nodes:
        if isinstance(node, VolumeSlice):
            return list(node.init_scale)

    extents = []
    for node in nodes:
        extent = getattr(node, 'data_extent', None)
        if extent is None:
            continue
        extent = np.asarray(extent, dtype=float)
        extent = extent[np.isfinite(extent)]
        if extent.size:
            extents.append(float(np.max(extent)))

    max_extent = max(extents) if extents else 0
    if max_extent <= 0:
        return [1.0, 1.0, 1.0]
    return [1.5 / max_extent] * 3


def _node_scale(init_scale, axis_scales):
    return [s * x for s, x in zip(init_scale, axis_scales)]


def _apply_node_scale(node, init_scale, axis_scales):
    if isinstance(node, VolumeSlice):
        node.update_scale(axis_scales)
    elif hasattr(node, 'scale'):
        node.scale = _node_scale(init_scale, axis_scales)


def update_clim(vmin, vmax, type, num, nodes):
    if vmin >= vmax:
        return
    for node in nodes:
        if type == 'bg':
            if hasattr(node, 'update_clim'):
                node.update_clim([vmin, vmax])
        elif type == 'fg':
            if hasattr(node, 'update_mask_clim'):
                node.update_mask_clim([vmin, vmax], num)

def update_cmap(cmap, nodes):
    for node in nodes:
        if hasattr(node, 'update_cmap'):
            if cmap=='pre-set':
                cmap = None
            node.update_cmap(cmap)

def update_mask_cmap(cmapname, alpha, excpt, num, first, nodes):
    if cmapname == 'pre-set':
        cmap = nodes[first]._fg_cmaps_preset[num]
    else:
        cmap = cmapname
    cmap = colormap.fast_set_cmap(cmap, alpha, excpt)

    for node in nodes:
        if hasattr(node, 'update_mask_cmap'):
            node.update_mask_cmap(cmap, num)



def _region2image(pts2d, server: viser.ViserServer):
    client = list(server.get_clients().values())[0]

    h, w = client.camera.image_height, client.camera.image_width

    # Convert normalized box to pixel coordinates
    (u0, v0), (u1, v1) = pts2d
    x0, y0 = int(u0 * w), int(v0 * h)
    x1, y1 = int(u1 * w), int(v1 * h)

    # Clamp + order
    lx, rx = sorted((max(0, x0), min(w, x1)))
    ly, ry = sorted((max(0, y0), min(h, y1)))

    # Create white canvas
    canvas = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Draw rectangle
    draw.rectangle([(lx, ly), (rx, ry)], outline=(255, 0, 255), width=2)

    # Draw filled circles at each corner (green)
    for px, py in [(lx, ly), (rx, ly), (rx, ry), (lx, ry)]:
        draw.ellipse([(px-4, py-4), (px+4, py+4)], fill=(0, 255, 0))

    return np.array(canvas)


class Server(viser.ViserServer):

    def __init__(self,
                 host: str = "0.0.0.0",
                 port: int = 8080,
                 label='cigvis-viser',
                 verbose: bool = False,
                 **kwargs):
        super().__init__(host, port, label, verbose, **kwargs)

        self.background_image = None
        self.draw_slices = -1
        self.nodes = None
        self._link_servers = []
        self.changed = False

    def _clear_plot_state(self):
        self.draw_slices = -1
        self.mask_num = 0
        for attr in (
            '_guix', '_guiy', '_guiz', '_guiclim', '_guicmap',
            '_maskclim1', '_maskcmap1', '_maskalpha1', '_maskexcpt1',
            '_maskclim2', '_maskcmap2', '_maskalpha2', '_maskexcpt2',
            '_maskclim3', '_maskcmap3', '_maskalpha3', '_maskexcpt3',
            '_gui_scale', '_region_left', '_region_right',
            '_gui_slice_handles',
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def link(self, server):
        if self.nodes is not None:
            raise RuntimeError("Please link a server before adding nodes")
        if type(server) is not Server:
            raise ValueError(f"only support `Server` class, but got: {type(server)}")
        self._link_servers += [server]

    def init_from_nodes(self, nodes, axis_scales, fov, look_at, wxyz, position):
        fov = fov * np.pi / 180
        if not nodes:
            raise ValueError("viserplot.plot3D requires at least one node")

        self._clear_plot_state()
        self.nodes = nodes
        self._axis_scales = tuple(
            axis_scales if axis_scales is not None else (1, 1, 1)
        )

        init_scale = _infer_init_scale(nodes)
        for i, node in enumerate(nodes):
            if isinstance(node, VolumeSlice):
                self.draw_slices = i

        self.init_scale = init_scale
        sliceid, meshid, logsid = 0, 0, 0
        for node in nodes:
            _apply_node_scale(node, init_scale, self._axis_scales)
            if isinstance(node, VolumeSlice):
                node.name = f'slice-{node.axis}-{sliceid}'
                sliceid += 1
            elif isinstance(node, MeshNode):
                node.name = f'mesh{meshid}'
                meshid += 1
            elif isinstance(node, LogBase):
                node.name = f'logs{logsid}-{node.base_name}'
                logsid += 1
            node.server = self
        
        if self.draw_slices >= 0:
            self.mask_num = len(nodes[self.draw_slices].masks)

        self._add_slices_gui()
        self._add_params_gui()
        self._add_screenshot_gui()
        self._add_state_gui()

        @self.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.fov = fov  # Or some other angle in radians, np.pi / 6 -> 30 degree
            if look_at is None:
                client.camera.look_at = (1, 1, 0)
            else:
                client.camera.look_at = tuple(look_at)
            if wxyz is not None:
                client.camera.wxyz = wxyz
            if position is not None:
                client.camera.position = tuple(position)
            # gui_camera.value = _fmt_camera_text(client.camera)

        self.scene.set_up_direction((0.0, 0.0, -1.0))

    def _add_slices_gui(self):
        # gui slices slibers to control slices position
        with self.gui.add_folder("slices pos"):
            self._gui_slice_handles = {}
            for axis, attr in (('x', '_guix'), ('y', '_guiy'), ('z', '_guiz')):
                axis_nodes = [
                    node for node in self.nodes
                    if isinstance(node, VolumeSlice) and node.axis == axis
                ]
                handles = []
                for idx, node in enumerate(axis_nodes):
                    label = axis if len(axis_nodes) == 1 else f'{axis}{idx}'
                    handle = self.gui.add_slider(
                        label,
                        min=0,
                        max=node.limit[1] - 1,
                        step=1,
                        initial_value=node.pos,
                    )
                    handle.on_update(
                        lambda _, node=node, handle=handle:
                        node.update_node(handle.value)
                    )
                    handles.append(handle)

                if handles:
                    setattr(self, attr, handles[0])
                    self._gui_slice_handles[axis] = handles

    def _add_params_gui(self):
        with self.gui.add_folder("parameters"):
            if self.draw_slices >= 0:
                step = (self.nodes[self.draw_slices].clim[1] - self.nodes[self.draw_slices].clim[0] + 1e-6) / 100
                self._guiclim = self.gui.add_vector2('clim', initial_value=tuple(self.nodes[self.draw_slices].clim), step=step)
                self._guiclim.on_update(lambda _: update_clim(*self._guiclim.value, 'bg', -1, nodes=self.nodes))

                self._guicmap = self.gui.add_dropdown(
                    'cmap',
                    options=[
                        'pre-set', 'gray', 'seismic', 'Petrel', 'stratum', 'jet', 'bwp'
                    ],
                    initial_value='pre-set',
                )
                self._guicmap.on_update(lambda _: update_cmap(self._guicmap.value, nodes=self.nodes))

                if self.mask_num > 0:
                    step1 = (self.nodes[self.draw_slices].fg_clims[0][1] - self.nodes[self.draw_slices].fg_clims[0][0] + 1e-6) / 100
                    self._maskclim1 = self.gui.add_vector2('mask_clim1', initial_value=tuple(self.nodes[self.draw_slices].fg_clims[0]), step=step1)
                    self._maskclim1.on_update(lambda _: update_clim(*self._maskclim1.value, 'fg', 0, nodes=self.nodes))
                    self._maskcmap1 = self.gui.add_dropdown('mask_cmap1', options=['pre-set', 'jet', 'stratum', 'Faults', 'gray'], initial_value='pre-set')
                    alpha1 = self.nodes[self.draw_slices].fg_cmaps[0](0.5)[-1]
                    self._maskalpha1 = self.gui.add_slider('mask_alpha1', min=0, max=1, step=0.05, initial_value=alpha1)
                    excpt1 = self.nodes[self.draw_slices].fg_cmaps[0].excpt if hasattr(self.nodes[self.draw_slices].fg_cmaps[0], 'excpt') else 'none'
                    self._maskexcpt1 = self.gui.add_dropdown('mask_excpt1', options=['none', 'min', 'max', 'ramp'], initial_value=excpt1)
                    self._maskcmap1.on_update(lambda _: update_mask_cmap(self._maskcmap1.value, self._maskalpha1.value, self._maskexcpt1.value, 0, self.draw_slices, nodes=self.nodes))
                    self._maskalpha1.on_update(lambda _: update_mask_cmap(self._maskcmap1.value, self._maskalpha1.value, self._maskexcpt1.value, 0, self.draw_slices, nodes=self.nodes))
                    self._maskexcpt1.on_update(lambda _: update_mask_cmap(self._maskcmap1.value, self._maskalpha1.value, self._maskexcpt1.value, 0, self.draw_slices, nodes=self.nodes))

                if self.mask_num > 1:
                    step2 = (self.nodes[self.draw_slices].fg_clims[1][1] - self.nodes[self.draw_slices].fg_clims[1][0] + 1e-6) / 100
                    self._maskclim2 = self.gui.add_vector2('mask_clim2', initial_value=tuple(self.nodes[self.draw_slices].fg_clims[1]), step=step2)
                    self._maskclim2.on_update(lambda _: update_clim(*self._maskclim2.value, 'fg', 1, nodes=self.nodes))
                    self._maskcmap2 = self.gui.add_dropdown('mask_cmap2', options=['pre-set', 'jet', 'stratum', 'Faults', 'gray'], initial_value='pre-set')
                    alpha2 = self.nodes[self.draw_slices].fg_cmaps[1](0.5)[-1]
                    self._maskalpha2 = self.gui.add_slider('mask_alpha2', min=0, max=1, step=0.05, initial_value=alpha2)
                    excpt2 = self.nodes[self.draw_slices].fg_cmaps[1].excpt if hasattr(self.nodes[self.draw_slices].fg_cmaps[1], 'excpt') else 'none'
                    self._maskexcpt2 = self.gui.add_dropdown('mask_excpt2', options=['none', 'min', 'max', 'ramp'], initial_value=excpt2)
                    self._maskcmap2.on_update(lambda _: update_mask_cmap(self._maskcmap2.value, self._maskalpha2.value, self._maskexcpt2.value, 1, self.draw_slices, nodes=self.nodes))
                    self._maskalpha2.on_update(lambda _: update_mask_cmap(self._maskcmap2.value, self._maskalpha2.value, self._maskexcpt2.value, 1, self.draw_slices, nodes=self.nodes))
                    self._maskexcpt2.on_update(lambda _: update_mask_cmap(self._maskcmap2.value, self._maskalpha2.value, self._maskexcpt2.value, 1, self.draw_slices, nodes=self.nodes))


                if self.mask_num > 2:
                    step3 = (self.nodes[self.draw_slices].fg_clims[2][1] - self.nodes[self.draw_slices].fg_clims[2][0] + 1e-6) / 100
                    self._maskclim3 = self.gui.add_vector2('mask_clim3', initial_value=tuple(self.nodes[self.draw_slices].fg_clims[2]), step=step3)
                    self._maskclim3.on_update(lambda _: update_clim(*self._maskclim3.value, 'fg', 2, nodes=self.nodes))
                    self._maskcmap3 = self.gui.add_dropdown('mask_cmap3', options=['pre-set', 'jet', 'stratum', 'Faults', 'gray'], initial_value='pre-set')
                    alpha3 = self.nodes[self.draw_slices].fg_cmaps[2](0.5)[-1]
                    self._maskalpha3 = self.gui.add_slider('mask_alpha3', min=0, max=1, step=0.05, initial_value=alpha3)
                    excpt3 = self.nodes[self.draw_slices].fg_cmaps[2].excpt if hasattr(self.nodes[self.draw_slices].fg_cmaps[2], 'excpt') else 'none'
                    self._maskexcpt3 = self.gui.add_dropdown('mask_excpt3', options=['none', 'min', 'max', 'ramp'], initial_value=excpt3)
                    self._maskcmap3.on_update(lambda _: update_mask_cmap(self._maskcmap3.value, self._maskalpha3.value, self._maskexcpt3.value, 2, self.draw_slices, nodes=self.nodes))
                    self._maskalpha3.on_update(lambda _: update_mask_cmap(self._maskcmap3.value, self._maskalpha3.value, self._maskexcpt3.value, 2, self.draw_slices, nodes=self.nodes))
                    self._maskexcpt3.on_update(lambda _: update_mask_cmap(self._maskcmap3.value, self._maskalpha3.value, self._maskexcpt3.value, 2, self.draw_slices, nodes=self.nodes))

            # gui to control aspect
            def _update_scale(scale, nodes):
                for node in nodes:
                    _apply_node_scale(node, self.init_scale, scale)

            self._gui_scale = self.gui.add_vector3('scale', initial_value=self._axis_scales, step=0.05, min=(0.1, 0.1, 0.1))
            self._gui_scale.on_update(lambda _: _update_scale(self._gui_scale.value, nodes=self.nodes))


    def _add_screenshot_gui(self):

        self._has_image_height = version.parse(viser.__version__) > version.parse("0.2.23")

        if not self._has_image_height:
            return

        with self.gui.add_folder("screenshot"):
            self._select_btn = self.gui.add_button("Select rectangular region")
            self._show_region = self.gui.add_checkbox("show boundary", False)
            self._region_left = self.gui.add_vector2("left_up", initial_value=(0, 0), min=(0, 0), max=(1, 1), step=0.001)
            self._region_right = self.gui.add_vector2("right_down", initial_value=(1, 1), min=(0, 0), max=(1, 1), step=0.001)
            self._render_btn = self.gui.add_button("Render and get PNG")


        def _select_region(server: viser.ViserServer):
            @server.scene.on_pointer_event(event_type="rect-select")
            def _box(event: viser.ScenePointerEvent) -> None:  # type: ignore[name-defined]
                # event.screen_pos is ((u_min, v_min), (u_max, v_max)), each in [0, 1]
                server._region_left.value = event.screen_pos[0]
                server._region_right.value = event.screen_pos[1]
                server.scene.remove_pointer_callback()

        def _draw_render_boundary(server: viser.ViserServer):
            region = (server._region_left.value, server._region_right.value)
            server.background_image = _region2image(region, server)
            if not server._show_region.value:
                return
            server.scene.set_background_image(server.background_image, format='png')

        def _update_box(server: viser.ViserServer):
            if not server._show_region.value:
                server.scene.set_background_image(None)
            else:
                if server.background_image is None:
                    region = (server._region_left.value, server._region_right.value)
                    server.background_image = _region2image(region, server)
                server.scene.set_background_image(server.background_image, format='png')

        def _render_save(server: viser.ViserServer):
            client = list(server.get_clients().values())[0]

            if server._show_region.value:
                server.scene.set_background_image(None)

            region = (server._region_left.value, server._region_right.value)
            # Full-res render at current camera resolution
            h, w = client.camera.image_height, client.camera.image_width
            image = client.get_render(height=h, width=w, transport_format='png')

            if server._show_region.value:
                server.scene.set_background_image(server.background_image)

            # Convert normalized box to pixel coordinates
            (u0, v0), (u1, v1) = region
            x0, y0 = int(u0 * w), int(v0 * h)
            x1, y1 = int(u1 * w), int(v1 * h)

            # Clamp + order
            x0, x1 = sorted((max(0, x0), min(w, x1)))
            y0, y1 = sorted((max(0, y0), min(h, y1)))

            if x1 <= x0 or y1 <= y0:
                print(f"[{client.client_id}] Empty crop; aborting.")
                return
            cropped = image[y0:y1, x0:x1]  # HWC
            # Encode & send
            png_bytes = iio.imwrite("<bytes>", cropped, extension=".png")
            client.send_file_download("selection.png", png_bytes)

        self._select_btn.on_click(lambda _: _select_region(self))
        self._region_left.on_update(lambda _: _draw_render_boundary(self))
        self._region_right.on_update(lambda _: _draw_render_boundary(self))
        self._show_region.on_update(lambda _: _update_box(self))
        self._render_btn.on_click(lambda _: _render_save(self))

    def _add_state_gui(self):
        with self.gui.add_folder("states"):
            self._gui_states = self.gui.add_button('print states')
            self._gui_states.on_click(lambda _: _print_states(self))
            if len(self._link_servers) > 0:
                self._sync = self.gui.add_button('synchronize')
                self._sync.on_click(lambda _: _update_states(self))

        def _update_states(srcserver):
            srcclient = list(srcserver.get_clients().values())[-1]
            for server in srcserver._link_servers:
                server: Server
                src_handles = getattr(srcserver, '_gui_slice_handles', {})
                dst_handles = getattr(server, '_gui_slice_handles', {})
                for axis in ('x', 'y', 'z'):
                    for src_handle, dst_handle in zip(
                        src_handles.get(axis, []),
                        dst_handles.get(axis, []),
                    ):
                        dst_handle.value = src_handle.value
                dstclient = list(server.get_clients().values())[-1]
                dstclient.camera.fov = srcclient.camera.fov
                dstclient.camera.look_at = srcclient.camera.look_at
                dstclient.camera.wxyz = srcclient.camera.wxyz
                dstclient.camera.position = srcclient.camera.position

    def reset(self):
        self.scene.reset()
        self.gui.reset()



def _print_states(server: Server):
    clients = list(server.get_clients().values())
    if not clients:
        print('')
        print('No connected viser client. Open the viewer before printing states.')
        print('')
        return

    client = clients[0]

    camera = client.camera
    print('')
    print('===== Copyable cigvis viser state =====')
    print("# Paste into cigvis.viserplot.plot3D(...):")
    scale_handle = getattr(server, '_gui_scale', None)
    scale_value = (
        scale_handle.value
        if scale_handle is not None
        else getattr(server, '_axis_scales', (1, 1, 1))
    )
    _print_kw('axis_scales', scale_value)
    _print_kw('fov', camera.fov * 180 / np.pi)
    _print_kw('look_at', camera.look_at)
    _print_kw('wxyz', camera.wxyz)
    _print_kw('position', camera.position)

    pos = {}
    slice_handles = getattr(server, '_gui_slice_handles', {})
    for axis, attr in (('x', '_guix'), ('y', '_guiy'), ('z', '_guiz')):
        handles = slice_handles.get(axis)
        if handles is None:
            handle = getattr(server, attr, None)
            handles = [] if handle is None else [handle]
        if handles:
            pos[axis] = [handle.value for handle in handles]
    if pos:
        print('')
        print("# Paste into cigvis.viserplot.create_slices(...):")
        _print_kw('pos', pos)
        if hasattr(server, '_guiclim'):
            _print_kw('clim', server._guiclim.value)
        cmap = getattr(getattr(server, '_guicmap', None), 'value', None)
        if cmap is not None and cmap != 'pre-set':
            _print_kw('cmap', cmap)
        elif cmap == 'pre-set':
            print("    # cmap is unchanged from the node preset.")

    mask_clims = []
    mask_cmaps = []
    mask_alpha = []
    mask_excpt = []
    for index in range(1, getattr(server, 'mask_num', 0) + 1):
        clim = getattr(server, f'_maskclim{index}', None)
        cmap = getattr(server, f'_maskcmap{index}', None)
        alpha = getattr(server, f'_maskalpha{index}', None)
        excpt = getattr(server, f'_maskexcpt{index}', None)
        if clim is not None:
            mask_clims.append(clim.value)
        if cmap is not None:
            mask_cmaps.append(cmap.value)
        if alpha is not None:
            mask_alpha.append(alpha.value)
        if excpt is not None:
            mask_excpt.append(None if excpt.value == 'none' else excpt.value)
    if mask_clims or mask_cmaps or mask_alpha or mask_excpt:
        print('')
        print("# Current mask GUI values:")
        if mask_clims:
            _print_kw('mask_clims', mask_clims)
        if mask_cmaps:
            _print_kw('mask_cmaps', mask_cmaps)
        if mask_alpha:
            _print_kw('mask_alpha', mask_alpha)
        if mask_excpt:
            _print_kw('mask_excpt', mask_excpt)

    if getattr(server, '_has_image_height', False):
        print('')
        print("# Current screenshot region:")
        _print_kw('left_up', server._region_left.value)
        _print_kw('right_down', server._region_right.value)

    print('')




def _round(f):
    if np.isscalar(f):
        return round(f, 2)
    if isinstance(f, list):
        return [round(x, 2) for x in f]
    if isinstance(f, np.ndarray):
        return np.round(f, 2)
