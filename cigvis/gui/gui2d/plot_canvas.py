# Copyright (c) 2024 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.

from pathlib import Path
from argparse import Namespace
from PyQt5 import QtWidgets as qtw

from cigvis.gui.custom_widgets import RectP

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from cigvis import colormap


class ImageMixin:

    def set_base_data(self, data):
        self.data = data
        self.plot()

    def set_cmap(self, cmap):
        try:
            self.params['cmap'] = colormap.get_cmap_from_str(cmap)
            if self.baseim:
                self.baseim.set_cmap(self.params['cmap'])
                self.draw()
        except Exception as e:
            qtw.QMessageBox.critical(self, "Error", f"Error colormap: {e}")

    def set_interp(self, interp):
        self.params['interpolation'] = interp
        if self.baseim:
            self.baseim.set_interpolation(interp)
            self.draw()

    def set_vmin(self, vmin):
        vmin = float(vmin)
        self.params['vmin'] = vmin
        vmax = self.params['vmax'] if 'vmax' in self.params else -vmin
        clim = [vmin, vmax]
        if self.baseim:
            self.baseim.set(clim=clim)
            self.draw()

    def set_vmax(self, vmax):
        vmax = float(vmax)
        self.params['vmax'] = vmax
        vmin = self.params['vmin'] if 'vmin' in self.params else -vmax
        clim = [vmin, vmax]
        if self.baseim:
            self.baseim.set(clim=clim)
            self.draw()


class MaskImageMixin:

    def set_mask_data(self, data):
        if len(self.mask_params) == len(self.masks):
            self.mask_params.append({
                'cmap': 'jet',
                'alpha': 0.5,
                'except': 'None',
                'plot': {
                    'interpolation': 'bilinear'
                }
            })
            cmap = self.set_mask_cmap('jet', 0.5, 'None')
            self.mask_params[-1]['plot']['cmap'] = cmap
        self.masks.append(data)

        im = self.axes.imshow(data.T, **self.mask_params[-1]['plot'])
        self.maskim.append(im)
        self.draw()

    def set_mask_params(self, params: list):
        idx, mode, value = params
        if idx < 0:
            return
        if mode == 'vmin':
            self.mask_params[idx]['plot']['vmin'] = value
            vmax = self.mask_params[idx]['plot'][
                'vmax'] if 'vmax' in self.mask_params[idx]['plot'] else None
            if vmax is not None:
                clim = [value, vmax]
                self.maskim[idx].set(clim=clim)
        elif mode == 'vmax':
            self.mask_params[idx]['plot']['vmax'] = value
            vmin = self.mask_params[idx]['plot'][
                'vmin'] if 'vmin' in self.mask_params[idx]['plot'] else None
            if vmin is not None:
                clim = [vmin, value]
                self.maskim[idx].set(clim=clim)
        elif mode == 'cmap':
            self.mask_params[idx]['cmap'] = value
            cmap = self.set_mask_cmap(
                value,
                self.mask_params[idx]['alpha'],
                self.mask_params[idx]['except'],
            )
            self.mask_params[idx]['plot']['cmap'] = cmap
            self.maskim[idx].set_cmap(cmap)

        elif mode == 'interp':
            self.mask_params[idx]['plot']['interpolation'] = value
            self.maskim[idx].set_interpolation(value)
        elif mode == 'alpha':
            self.mask_params[idx]['alpha'] = value
            cmap = self.set_mask_cmap(
                self.mask_params[idx]['cmap'],
                value,
                self.mask_params[idx]['except'],
            )
            self.mask_params[idx]['plot']['cmap'] = cmap
            self.maskim[-1].set_cmap(cmap)
        elif mode == 'except':
            self.mask_params[idx]['except'] = value
            cmap = self.set_mask_cmap(
                self.mask_params[idx]['cmap'],
                self.mask_params[idx]['alpha'],
                value,
            )
            self.mask_params[idx]['plot']['cmap'] = cmap
            self.maskim[-1].set_cmap(cmap)

        self.draw()

    def set_mask_cmap(self, cmap, alpha, excpt: str):
        if excpt == 'None':
            cmap = colormap.set_alpha(cmap, alpha, False)
        elif excpt == 'min':
            cmap = colormap.set_alpha_except_min(cmap, alpha, False)
        elif excpt == 'max':
            cmap = colormap.set_alpha_except_max(cmap, alpha, False)
        # elif excpt.startswith('blow'):
        #     l = excpt[5:-1]
        #     try:
        #         l = float(l)
        #     except:
        #         qtw.QMessageBox.critical(self, "Warn", f"{excpt} not valid")
        #     cmap = colormap.set_alpha_except_bottom(cmap, alpha, )

        return cmap

    def remove_mask(self, idx):
        pass


class AnnotationMixin:

    def set_marker_mode(self, mode: int):
        self.marker_mode = mode

    def set_box_mode(self, mode: int):
        self.box_mode = mode

    def set_brush_mode(self, mode: int):
        self.brush_mode = mode

    def set_brush_size(self, size: int):
        self.brush_size = size

    def draw_marker(self, x, y):
        c = '#74ddd0' if self.marker_mode == 2 else '#f5c2cb'
        self.axes.plot(x, y, marker='o', color=c)
        self.draw()

    def on_mouse_press(self, event):
        if self.marker_mode >= 0 and event.inaxes:
            c = '#74ddd0' if self.marker_mode == 1 else '#f5c2cb'
            im = self.axes.plot(event.xdata, event.ydata, marker='o', color=c)
            self.marker_im.append(im[0])
            self.draw()
            self.marker_list.append(
                [event.xdata, event.ydata, self.marker_mode - 1])

        elif self.box_mode > 0 and event.inaxes:
            self.box_list.append(RectP())
            self.box_list[-1].add_p0event(event)
            self.rect = Rectangle((event.xdata, event.ydata),
                                  0,
                                  0,
                                  fill=False,
                                  color='white')
            patch = self.axes.add_patch(self.rect)
            self.box_im.append(patch)

            self.draw()

        elif self.brush_mode > 0 and event.inaxes:
            self.scribing = True
            self.scrib_list.append([[event.xdata], [event.ydata]])
            self.scribim_list.append(
                self.axes.plot(self.scrib_list[-1][0],
                               self.scrib_list[-1][1],
                               color='red',
                               linewidth=self.brush_size / 10)[0])
            self.draw()

    def on_mouse_move(self, event):
        if self.box_mode > 0 and self.rect is not None and event.inaxes:
            self.box_list[-1].add_p1event(event)
            self.rect.set_width(self.box_list[-1].x1 - self.box_list[-1].x0)
            self.rect.set_height(self.box_list[-1].y1 - self.box_list[-1].y0)
            self.rect.set_xy((self.box_list[-1].x0, self.box_list[-1].y0))
            self.draw()
        elif self.brush_mode > 0 and self.scribing and event.inaxes:
            self.scrib_list[-1][0].append(event.xdata)
            self.scrib_list[-1][1].append(event.ydata)
            self.scribim_list[-1].set_data(self.scrib_list[-1][0],
                                           self.scrib_list[-1][1])
            self.draw()

    def on_mouse_release(self, event):
        if self.box_mode > 0 and event.inaxes:
            self.rect = None
        elif self.brush_mode > 0 and event.inaxes:
            # self.scrib_list.append(self.scribp)
            self.scribing = False
            # self.draw()

    def on_leave(self, event):
        if self.box_mode > 0 or self.brush_mode > 0:
            evt = Namespace()
            evt.inaxes = True
            self.on_mouse_release(evt)

    def marker_undo(self):
        if len(self.marker_list) == 0:
            return
        p = self.marker_list.pop()
        im = self.marker_im.pop()
        im.remove()
        self.draw()

    def box_undo(self):
        if len(self.box_list) == 0:
            return
        p = self.box_list.pop()
        im = self.box_im.pop()
        im.remove()
        self.draw()

    def brush_undo(self):
        if len(self.scrib_list) == 0:
            return
        p = self.scrib_list.pop()
        im = self.scribim_list.pop()
        im.remove()
        self.draw()

    def marker_reset(self):
        self.marker_list.clear()
        for im in self.marker_im:
            im.remove()
        self.marker_im.clear()
        self.draw()

    def box_reset(self):
        self.box_list.clear()
        for im in self.box_im:
            im.remove()
        self.box_im.clear()
        self.draw()

    def brush_reset(self):
        self.scrib_list.clear()
        for im in self.scribim_list:
            im.remove()
        self.scribim_list.clear()
        self.draw()


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


class PlotCanvas(FigureCanvas, DraggableMixin, ImageMixin, AnnotationMixin,
                 MaskImageMixin):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_axis_off()
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        super().__init__(self.fig)
        self.setParent(parent)
        self.controlP = self.parent().controlP
        self.gstates = self.parent().gstates
        self.enableDragging()
        self.mpl_connect('button_press_event', self.on_mouse_press)
        self.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.mpl_connect('button_release_event', self.on_mouse_release)
        self.leaveEvent = self.on_leave
        self.init_states()

    def init_states(self):
        self.data = None
        self.params = {'cmap': 'gray', 'interpolation': 'bilinear'}
        self.baseim = None

        self.marker_mode = -1
        self.marker_list = []  # to save point number
        self.marker_im = []  # to save plt result

        self.box_mode = -1
        self.rect = None  # temp
        self.box_list = []  # to save rectPs
        self.box_im = []  # to save plt result

        self.brush_mode = -1
        self.scribing = False
        self.scrib_list = []
        self.scribim_list = []
        self.brush_size = 10

        # for mask
        self.masks = []  # ArrayLike
        self.maskim = []  # imshow
        self.mask_params = []  # params

    def set_data(self, data: np.ndarray):
        if self.gstates.loadType == 'base':
            self.set_base_data(data)
        else:
            self.set_mask_data(data)

    def plot(self):
        self.axes.clear()
        if self.data is None:
            self.axes.axis('off')
            self.baseim = None
        else:
            self.baseim = self.axes.imshow(self.data.T, **self.params)
        self.draw()

    def clear(self):
        self.data = None
        self.params = {'cmap': 'gray', 'interpolation': 'bilinear'}
        self.marker_reset()
        self.box_reset()
        self.brush_reset()
        self.brush_mode, self.box_mode, self.marker_mode = -1, -1, -1
        self.brush_size = 10
        self.plot()

    def save_fig(self):
        if not self.baseim:
            qtw.QMessageBox.critical(self, "Warn", "No Image")
            return

        fileName, _ = qtw.QFileDialog.getSaveFileName(
            self, "Save name", "",
            "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)")

        if fileName:
            self.fig.savefig(fileName,
                             bbox_inches='tight',
                             pad_inches=0.01,
                             dpi=300)
