# Copyright (c) 2024 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.

from pathlib import Path
from cigvis.vispynodes import VisCanvas, AxisAlignedImage
import cigvis
from cigvis import colormap
from PyQt5 import QtWidgets as qtw


class ImageMixin:

    def set_base_data(self, data):
        if self.data is None:
            self.data = data
            self.nodes = cigvis.create_slices(self.data, **self.params)
            self.canvas.add_nodes(self.nodes)
        else:
            self.clear()
            self.set_base_data(data)

    def set_cmap(self, cmap: str):
        try:
            self.params['cmap'] = colormap.cmap_to_vispy(cmap)
            if self.data is not None:
                self.set_attrs('cmap', self.params['cmap'])
        except Exception as e:
            qtw.QMessageBox.critical(self, "Error", f"Error colormap: {e}")

    def set_vmin(self, vmin: str):
        vmin = float(vmin)
        self.params['vmin'] = vmin
        if 'clim' in self.params:
            v1, v2 = self.params['clim']
            self.params['clim'] = [vmin, v2]
            self.set_attrs('clim', [vmin, v2])
        elif 'vmax' in self.params:
            clim = [vmin, self.params['vmax']]
            self.params['clim'] = clim
            self.set_attrs('clim', clim)

    def set_vmax(self, vmax: str):
        vmax = float(vmax)
        self.params['vmax'] = vmax
        if 'clim' in self.params:
            v1, v2 = self.params['clim']
            self.params['clim'] = [v1, vmax]
            self.set_attrs('clim', [v1, vmax])
        elif 'vmin' in self.params:
            clim = [self.params['vmin'], vmax]
            self.params['clim'] = clim
            self.set_attrs('clim', clim)

    def set_interp(self, interp: str):
        self.params['interpolation'] = interp
        self.set_attrs('interpolation', interp)


class MaskImageMixin:

    def set_mask_data(self, data):
        if len(self.mask_params) == len(self.masks):
            self.mask_params.append({
                'cmaps': 'jet',
                'interpolation': 'nearest',
                'alpha': 0.5,
                'excpt': 'None',
            })
        self.masks.append(data)
        self.nodes = cigvis.add_mask(self.nodes, data, **self.mask_params[-1])

    def set_mask_params(self, params: list):
        idx, mode, value = params
        if idx < 0 and len(self.mask_params) == len(self.masks):
            self.mask_params.append({
                'cmaps': 'jet',
                'interpolation': 'nearest',
                'alpha': 0.5,
                'excpt': 'None',
            })

        if mode == 'vmin':
            vmin = float(value)
            self.mask_params[idx]['vmin'] = vmin
            if 'clim' in self.mask_params[idx]:
                v1, v2 = self.mask_params[idx]['clim']
                self.mask_params[idx]['clim'] = [vmin, v2]
            elif 'vmax' in self.mask_params[idx]:
                clim = [vmin, self.mask_params[idx]['vmax']]
                self.mask_params[idx]['clim'] = clim
            if len(self.mask_params) == len(self.masks):
                self.set_mask_attrs('clim', self.mask_params[idx]['clim'], idx)

        elif mode == 'vmax':
            vmax = float(value)
            self.mask_params[idx]['vmax'] = vmax
            if 'clim' in self.mask_params[idx]:
                v1, v2 = self.mask_params[idx]['clim']
                self.mask_params[idx]['clim'] = [v1, vmax]
            elif 'vmin' in self.mask_params[idx]:
                clim = [self.mask_params[idx]['vmin'], vmax]
                self.mask_params[idx]['clim'] = clim
            if len(self.mask_params) == len(self.masks):
                self.set_mask_attrs('clim', self.mask_params[idx]['clim'], idx)

        elif mode == 'interp':
            self.mask_params[idx]['interp'] = value
            if len(self.mask_params) == len(self.masks):
                self.set_mask_attrs('interpolation',
                                    self.mask_params[idx]['interp'], idx)

        elif mode == 'cmap':
            cmap = self.set_mask_cmap(value, self.mask_params[idx]['alpha'],
                                      self.mask_params[idx]['excpt'])
            if cmap:
                self.mask_params[idx]['cmaps'] = cmap
            if len(self.mask_params) == len(self.masks):
                self.set_mask_attrs('cmap', self.mask_params[idx]['cmaps'],
                                    idx)

        elif mode == 'alpha':
            self.mask_params[idx]['alpha'] = float(value)
            cmap = self.set_mask_cmap(self.mask_params[idx]['cmaps'], value,
                                      self.mask_params[idx]['excpt'])
            if cmap:
                self.mask_params[idx]['cmaps'] = cmap
            if len(self.mask_params) == len(self.masks):
                self.set_mask_attrs('cmap', self.mask_params[idx]['cmaps'],
                                    idx)

        elif mode == 'except':
            self.mask_params[idx]['excpt'] = value
            cmap = self.set_mask_cmap(self.mask_params[idx]['cmaps'],
                                      self.mask_params[idx]['alpha'], value)
            if cmap:
                self.mask_params[idx]['cmaps'] = cmap
            if len(self.mask_params) == len(self.masks):
                self.set_mask_attrs('cmap', self.mask_params[idx]['cmaps'],
                                    idx)

    def set_mask_cmap(self, cmap, alpha, excpt: str = 'None'):
        try:
            if excpt == 'None':
                cmap = colormap.set_alpha(cmap, alpha)
            elif excpt == 'min':
                cmap = colormap.set_alpha_except_min(cmap, alpha)
            elif excpt == 'max':
                cmap = colormap.set_alpha_except_max(cmap, alpha)
            else:
                qtw.QMessageBox.critical(
                    self, "Error",
                    f"The except mode: {excpt} is not supported now")
                return

        except Exception as e:
            qtw.QMessageBox.critical(self, "Error", f"Error colormap: {e}")
            return

        return cmap

    def remove_mask(self, idx):
        for node in self.nodes:
            if isinstance(node, AxisAlignedImage):
                node.remove_mask(idx + 1)

    def mask_clear(self):
        for i in range(len(self.masks)):
            self.remove_mask(0)

        # clear masks
        for d in self.masks:
            del d
        self.masks.clear()
        self.mask_params.clear()


class CameraMixin:

    def set_azimuth(self, azimuth):
        if not hasattr(self.canvas, 'view'):
            return
        for view in self.canvas.view:
            view.camera.azimuth = azimuth

    def set_elevation(self, elevation):
        if not hasattr(self.canvas, 'view'):
            return
        for view in self.canvas.view:
            view.camera.elevation = elevation

    def set_fov(self, fov):
        if not hasattr(self.canvas, 'view'):
            return
        for view in self.canvas.view:
            view.camera.fov = fov

    def set_xpos(self, xpos):
        if xpos < 0 and len(self.nodes) == 0:
            return
        xpos = int(xpos)
        for node in self.nodes:
            if isinstance(node, AxisAlignedImage):
                if node.axis == 'x':
                    node._update_location(xpos)

    def set_ypos(self, ypos):
        if ypos < 0 and len(self.nodes) == 0:
            return
        ypos = int(ypos)
        for node in self.nodes:
            if isinstance(node, AxisAlignedImage):
                if node.axis == 'y':
                    node._update_location(ypos)

    def set_zpos(self, zpos):
        if zpos < 0 and len(self.nodes) == 0:
            return
        zpos = int(zpos)
        for node in self.nodes:
            if isinstance(node, AxisAlignedImage):
                if node.axis == 'z':
                    node._update_location(zpos)

    def set_aspectx(self, aspx):
        r = cigvis.is_x_reversed()
        aspx *= (1 - 2 * r)
        if not hasattr(self.canvas, 'view'):
            return
        for view in self.canvas.view:
            axis_scales = view.camera._flip_factors
            axis_scales[0] = aspx
            view.camera._flip_factors = axis_scales
            view.camera._update_camera_pos()
            self.canvas.update()

    def set_aspecty(self, aspy):
        r = cigvis.is_y_reversed()
        aspy *= (1 - 2 * r)
        if not hasattr(self.canvas, 'view'):
            return
        for view in self.canvas.view:
            axis_scales = view.camera._flip_factors
            axis_scales[1] = aspy
            view.camera._flip_factors = axis_scales
            view.camera._update_camera_pos()
            self.canvas.update()

    def set_aspectz(self, aspz):
        r = cigvis.is_z_reversed()
        aspz *= (1 - 2 * r)
        if not hasattr(self.canvas, 'view'):
            return
        for view in self.canvas.view:
            axis_scales = view.camera._flip_factors
            axis_scales[2] = aspz
            view.camera._flip_factors = axis_scales
            view.camera._update_camera_pos()
            self.canvas.update()

    def get_params(self):
        out = None
        if hasattr(self.canvas, 'view'):
            camera = self.canvas.view[0].camera
            out = [camera.azimuth, camera.elevation, camera.fov]
            out = list(map(int, out))

        if len(self.nodes) > 0:
            pos = {'x': [], 'y': [], 'z': []}
            for node in self.nodes:
                if isinstance(node, AxisAlignedImage):
                    axis = node.axis
                    apos = node.pos
                    pos[axis].append(apos)

            out.append(pos)

        return out


class DraggableMixin:

    def enableDragging(self):
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        filePath = event.mimeData().urls()[0].toLocalFile()
        if Path(filePath).is_file():
            self.controlP.loadBtn.loadData(filePath)
        elif Path(filePath).is_dir():
            self.controlP.loadfolder.loadFd.loadFolder(filePath)

    def handleFileDropped(self):
        raise NotImplementedError("Need Implemented in main class")


class PlotCanvas(qtw.QWidget, DraggableMixin, CameraMixin, ImageMixin,
                 MaskImageMixin):

    def __init__(self, *args, parent=None, **kwargs):
        super().__init__(parent)
        self.layout = qtw.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)  # 减小边界

        self.canvas = CanvasWrapper(*args, **kwargs)
        self.canvas.create_native()
        self.canvas.native.setParent(self)
        self.layout.addWidget(self.canvas.native)

        self.controlP = self.parent().controlP
        self.gstates = self.parent().gstates

        self.enableDragging()
        self.init_states()

    def init_states(self):
        self.data = None
        self.nodes = []
        self.params = {'cmap': 'gray', 'interpolation': 'bilinear'}

        self.masks = []
        self.mask_params = []

    def set_data(self, data):
        if self.gstates.loadType == 'base':
            self.set_base_data(data)
        else:
            self.set_mask_data(data)

    def set_attrs(self, name, value, types=AxisAlignedImage):
        for node in self.nodes:
            if isinstance(node, types):
                setattr(node, name, value)

    def set_mask_attrs(self, name, value, idx, types=AxisAlignedImage):
        for node in self.nodes:
            if isinstance(node, types):
                setattr(node.overlaid_images[idx + 1], name, value)

    def clear(self):
        self.mask_clear()

        # clear nodes
        for node in self.nodes:
            node.parent = None
            del node
        self.nodes.clear()

        # clear data
        del self.data
        self.data = None

        # init
        self.params = {'cmap': 'gray', 'interpolation': 'bilinear'}


class CanvasWrapper(VisCanvas):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
