# Copyright (c) 2025 Jintao Li.
# University of Science and Technology of China (USTC).
# All rights reserved.

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

    def link(self, server):
        if self.nodes is not None:
            raise RuntimeError("Please link a server before adding nodes")
        if type(server) is not Server:
            raise ValueError(f"only support `Server` class, but got: {type(server)}")
        self._link_servers += [server]

    def init_from_nodes(self, nodes, axis_scales, fov, look_at, wxyz, position):
        fov = fov * np.pi / 180
        init_scale = -1
        self.nodes = nodes
        for i, node in enumerate(nodes):
            if isinstance(node, VolumeSlice):
                init_scale = node.init_scale
                node.update_scale(axis_scales)
                self.draw_slices = i

        if init_scale == -1:  # no slices # TODO: for other types, Well logs?
            init_scale = 100
            for node in nodes:
                if isinstance(node, MeshNode):
                    init_scale = min(min(node.scale), init_scale)
            init_scale = [init_scale] * 3

        self.init_scale = init_scale
        # update scale of meshes
        meshid, logsid = 0, 0
        for node in nodes:
            if isinstance(node, MeshNode):
                node.scale = [s * x for s, x in zip(init_scale, axis_scales)]
                node.name = f'mesh{meshid}'
                meshid += 1
            elif isinstance(node, LogBase):
                node.scale = [s * x for s, x in zip(init_scale, axis_scales)]
                node.name = f'logs{logsid}-{node.base_name}'
                logsid += 1
            node.server = self
        
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
            nodex = [
                node for node in self.nodes
                if isinstance(node, VolumeSlice) and node.axis == 'x'
            ]
            nodey = [
                node for node in self.nodes
                if isinstance(node, VolumeSlice) and node.axis == 'y'
            ]
            nodez = [
                node for node in self.nodes
                if isinstance(node, VolumeSlice) and node.axis == 'z'
            ]
            if len(nodex) > 0:
                nodex = nodex[0]
                self._guix = self.gui.add_slider(
                    'x',
                    min=0,
                    max=nodex.limit[1] - 1,
                    step=1,
                    initial_value=nodex.pos,
                )
                self._guix.on_update(lambda _: nodex.update_node(self._guix.value))

            if len(nodey) > 0:
                nodey = nodey[0]
                self._guiy = self.gui.add_slider(
                    'y',
                    min=0,
                    max=nodey.limit[1] - 1,
                    step=1,
                    initial_value=nodey.pos,
                )
                self._guiy.on_update(lambda _: nodey.update_node(self._guiy.value))

            if len(nodez) > 0:
                nodez = nodez[0]
                self._guiz = self.gui.add_slider(
                    'z',
                    min=0,
                    max=nodez.limit[1] - 1,
                    step=1,
                    initial_value=nodez.pos,
                )
                self._guiz.on_update(lambda _: nodez.update_node(self._guiz.value))

    def _add_params_gui(self):
        with self.gui.add_folder("paramters"):
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
                    self._maskclim2 = self.gui.add_vector2('maskclim2', initial_value=tuple(self.nodes[self.draw_slices].fg_clims[1]), step=step2)
                    self._maskclim2.on_update(lambda _: update_clim(*self._maskclim2.value, 'fg', 1, nodes=self.nodes))
                    self._maskcmap2 = self.gui.add_dropdown('mask_cmap2', options=['pre-set', 'jet', 'stratum', 'Faults', 'gray'], initial_value='pre-set')
                    alpha2 = self.nodes[self.draw_slices].fg_cmaps[1](0.5)[-1]
                    self._maskalpha2 = self.gui.add_slider('mask_alpha2', min=0, max=1, step=0.05, initial_value=alpha2)
                    excpt2 = self.nodes[self.draw_slices].fg_cmaps[1].excpt if hasattr(self.nodes[self.draw_slices].fg_cmaps[1], 'excpt') else 'none'
                    maskexcpt2 = self.gui.add_dropdown('mask_excpt2', options=['none', 'min', 'max', 'ramp'], initial_value=excpt2)
                    self._maskcmap2.on_update(lambda _: update_mask_cmap(self._maskcmap2.value, self._maskalpha2.value, self._maskexcpt2.value, 1, self.draw_slices, nodes=self.nodes))
                    self._maskalpha2.on_update(lambda _: update_mask_cmap(self._maskcmap2.value, self._maskalpha2.value, self._maskexcpt2.value, 1, self.draw_slices, nodes=self.nodes))
                    self._maskexcpt2.on_update(lambda _: update_mask_cmap(self._maskcmap2.value, self._maskalpha2.value, self._maskexcpt2.value, 1, self.draw_slices, nodes=self.nodes))


                if self.mask_num > 2:
                    step3 = (self.nodes[self.draw_slices].fg_clims[2][1] - self.nodes[self.draw_slices].fg_clims[2][0] + 1e-6) / 100
                    self._maskclim3 = self.gui.add_vector2('maskclim3', initial_value=tuple(self.nodes[self.draw_slices].fg_clims[2]), step=step3)
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
                    if isinstance(node, VolumeSlice):
                        node.update_scale(scale)
                    elif isinstance(node, MeshNode):
                        node.scale = [s * x for s, x in zip(self.init_scale, scale)]

            self._gui_scale = self.gui.add_vector3('scale', initial_value=(1, 1, 1), step=0.05, min=(0.1, 0.1, 0.1))
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
                if hasattr(srcserver, "_guix") and hasattr(server, "_guix"):
                    server._guix.value = self._guix.value
                if hasattr(srcserver, "_guiy") and hasattr(server, "_guiy"):
                    server._guiy.value = self._guiy.value
                if hasattr(srcserver, "_guiz") and hasattr(server, "_guiz"):
                    server._guiz.value = self._guiz.value
                dstclient = list(server.get_clients().values())[-1]
                dstclient.camera.fov = srcclient.camera.fov
                dstclient.camera.look_at = srcclient.camera.look_at
                dstclient.camera.wxyz = srcclient.camera.wxyz
                dstclient.camera.position = srcclient.camera.position

    def reset(self):
        self.scene.reset()
        self.gui.reset()



def _print_states(server: Server):
    client = list(server.get_clients().values())[0]

    camera = client.camera
    print('')
    print('----------- Current States ------------')
    print(f'fov: {_round(camera.fov * 180 / np.pi)}')
    print(f'look_at: {_round(camera.look_at)}')
    print(f'wxyz: {_round(camera.wxyz)}')
    print(f'position: {_round(camera.position)}')
    print('----------- axis position -------------')
    print(f'x: {server._guix.value}, y: {server._guiy.value}, z: {server._guiz.value}')
    print('----------- parameters ----------------')
    print(f'cmap: {server._guicmap.value}')
    print(f'clim: {server._guiclim.value}')
    if server.mask_num > 0:
        print(f'mask_clim1: {server._maskclim1.value}')
        print(f'mask_cmap1: {server._maskcmap1.value}')
        print(f'mask_alpha1: {server._maskalpha1.value}')
        print(f'mask_excpt1: {server._maskexcpt1.value}')
    if server.mask_num > 1:
        print(f'mask_clim2: {server._maskclim2.value}')
        print(f'mask_cmap2: {server._maskcmap2.value}')
        print(f'mask_alpha2: {server._maskalpha2.value}')
        print(f'mask_excpt2: {server._maskexcpt2.value}')
    if server.mask_num > 2:
        print(f'mask_clim3: {server._maskclim3.value}')
        print(f'mask_cmap3: {server._maskcmap3.value}')
        print(f'mask_alpha3: {server._maskalpha3.value}')
        print(f'mask_excpt3: {server._maskexcpt3.value}')
    print('----------- Aspect Ratio --------------')
    print(f'scale: {server._gui_scale.value}') # yapf: disable
    if server._has_image_height:
        print('----------- Screenshot Region --------------')
        print(f'left_up: {server._region_left.value}')
        print(f'right_down: {server._region_right.value}')

    print('')




def _round(f):
    if np.isscalar(f):
        return round(f, 2)
    if isinstance(f, list):
        return [round(x, 2) for x in f]
    if isinstance(f, np.ndarray):
        return np.round(f, 2)