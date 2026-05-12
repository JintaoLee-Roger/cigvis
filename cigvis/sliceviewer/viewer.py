# Copyright (c) 2025 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
"""Panel+Plotly rendering layer for sliceviewer."""

import html
import math
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np

from .nodes import SliceNode

try:
    import plotly.graph_objects as go
    from plotly.basedatatypes import BaseTraceType as _BaseTraceType
    import panel as pn
except ImportError as e:
    raise ImportError(
        "SliceViewer requires panel and plotly.\n"
        "Install: pip install panel plotly anywidget"
    ) from e

_CMAPS = ['gray', 'RdBu', 'seismic', 'Petrel', 'jet', 'Viridis', 'Greys', 'Cividis']
_DEFAULT_ADDRESS = 'localhost'
_RAW_CSS = """
html, body {
  margin: 0;
  background: #eef2f6;
  color: #1f2937;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.sliceviewer-app {
  box-sizing: border-box;
  padding: 12px;
  gap: 12px;
  background: #e8edf3;
}
.sliceviewer-sidebar {
  box-sizing: border-box;
  padding: 12px;
  border-radius: 8px;
  background: #f8fafc;
  border: 1px solid #cbd5e1;
  box-shadow: 0 8px 20px rgba(15, 23, 42, 0.10);
}
.sliceviewer-sidebar .bk-input,
.sliceviewer-sidebar select,
.sliceviewer-sidebar input {
  background-color: #ffffff !important;
  color: #0f172a !important;
  border-color: #94a3b8 !important;
  border-radius: 8px !important;
}
.sliceviewer-sidebar label,
.sliceviewer-sidebar .bk-input-group label {
  color: #334155 !important;
  font-weight: 650 !important;
}
.sliceviewer-title {
  color: #0f172a;
  font-size: 18px;
  font-weight: 760;
  line-height: 1.1;
  letter-spacing: 0;
  margin: 0;
}
.sliceviewer-subtitle {
  color: #64748b;
  font-size: 11px;
  font-weight: 650;
  letter-spacing: .08em;
  margin-top: 6px;
  text-transform: uppercase;
}
.sliceviewer-section {
  margin-top: 10px;
}
.sliceviewer-section-title {
  color: #0369a1;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: .08em;
  margin-bottom: 5px;
  text-transform: uppercase;
}
.sliceviewer-fixed-range {
  color: #64748b;
  font-size: 11px;
  font-weight: 500;
}
.sliceviewer-plot-card {
  box-sizing: border-box;
  padding: 8px;
  border-radius: 8px;
  background: #ffffff;
  border: 1px solid #e5e7eb;
  box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
}
"""


class SliceViewerServer:
    """A persistent Panel server. Create once, push content multiple times."""

    def __init__(self, port: int = 5007, address: str = _DEFAULT_ADDRESS):
        _load_extension()
        self.port = port
        self.address = address
        self._holder = pn.Column(sizing_mode='stretch_width')
        self._server = pn.serve(
            self._holder,
            port=port,
            address=address,
            show=False,
            threaded=True,
            websocket_origin=_ws_origins(address, port),
        )

    def _update(self, layout):
        self._holder[:] = [layout]

    def __repr__(self):
        return f"SliceViewerServer(port={self.port}, address='{self.address}')"


def create_server(port: int = 5007,
                  address: str = _DEFAULT_ADDRESS) -> SliceViewerServer:
    """Create a persistent Panel server."""
    return SliceViewerServer(port=port, address=address)


def link(nodes1: List, nodes2: List):
    """Explicitly link two node lists for comparison layouts."""
    sn1 = _get_slice_node(nodes1)
    sn2 = _get_slice_node(nodes2)
    if sn1 is None or sn2 is None:
        raise ValueError("Both node lists must contain a SliceNode (from create_slice)")
    sn1._linked = sn2
    sn2._linked = sn1


def build_layout(
    nodes: Union[List, List[List]],
    title: str = '',
    cbar_label: str = 'Amplitude',
    grid: Optional[Tuple[int, int]] = None,
):
    """Build the Panel layout without starting or updating a server."""
    _load_extension()

    if nodes and isinstance(nodes[0], list):
        return _build_compare_layout(nodes, title, cbar_label, grid)
    return _build_single_layout(nodes, title, cbar_label)


def show(
    nodes: Union[List, List[List]],
    server: Optional[SliceViewerServer] = None,
    port: int = 5007,
    address: str = _DEFAULT_ADDRESS,
    title: str = '',
    cbar_label: str = 'Amplitude',
    grid: Optional[Tuple[int, int]] = None,
    launch: bool = True,
    **kwargs,
):
    """Build a Panel+Plotly interactive slice viewer."""
    layout = build_layout(nodes, title=title, cbar_label=cbar_label, grid=grid)

    if server is not None:
        server._update(layout)
    elif launch:
        layout.show(port=port, address=address,
                    websocket_origin=_ws_origins(address, port), **kwargs)

    return layout


def _build_single_layout(nodes: List, title: str, cbar_label: str):
    entry = _make_entry(nodes, title)
    controls = _make_controls([entry], title, 'single volume')
    return _make_app_layout(controls, [entry['pane']], grid=None)


def _build_compare_layout(
    node_lists: List[List],
    title: str,
    cbar_label: str,
    grid: Optional[Tuple[int, int]],
):
    entries = [_make_entry(nodes, title) for nodes in node_lists]
    if not entries:
        raise ValueError("No SliceNodes found in node lists")

    shapes = [entry['sn'].shape for entry in entries]
    if len(set(shapes)) > 1:
        warnings.warn(
            "sliceviewer comparison received different shapes; shared controls "
            "will clamp indices to each volume's valid range.",
            UserWarning,
            stacklevel=2,
        )

    controls = _make_controls(entries, title, f'{len(entries)} volumes')
    return _make_app_layout(
        controls,
        [entry['pane'] for entry in entries],
        grid=grid,
    )


def _make_entry(nodes: List, title: str):
    sn = _get_slice_node(nodes)
    if sn is None:
        raise ValueError("nodes must contain at least one SliceNode (from create_slice)")
    anno = [n for n in nodes if isinstance(n, _BaseTraceType)]
    return {
        'sn': sn,
        'anno': anno,
        'pane': _make_plot_pane(sn, anno, title),
    }


def _make_controls(entries: List[dict], title: str, subtitle: str):
    sns = [entry['sn'] for entry in entries]
    base = sns[0]
    sync = {'axes': False, 'clim': False, 'aspect': False, 'axis_options': False}

    y_select = pn.widgets.Select(
        name='Y axis',
        options=_axis_options(base, exclude=base.x_axis),
        value=base.y_axis,
        width=118,
    )
    x_select = pn.widgets.Select(
        name='X axis',
        options=_axis_options(base, exclude=base.y_axis),
        value=base.x_axis,
        width=118,
    )
    hidden_controls = pn.Column(width=248)

    vmin_w = pn.widgets.FloatInput(
        name='vmin',
        value=_nice_float(base.clim[0]),
        width=118,
    )
    vmax_w = pn.widgets.FloatInput(
        name='vmax',
        value=_nice_float(base.clim[1]),
        width=118,
    )
    auto_btn = pn.widgets.Button(name='Auto clim', button_type='primary', width=248)
    cmap_sel = pn.widgets.Select(
        name='Colormap',
        options=_CMAPS,
        value=_cmap_name(base.cmap),
        width=248,
    )
    aspect_mode = pn.widgets.Select(
        name='Aspect',
        options={'Equal': 'equal', 'Free': 'auto', 'Custom': 'custom'},
        value=_aspect_mode(base.aspect),
        width=118,
    )
    aspect_value = pn.widgets.FloatInput(
        name='Y/X',
        value=_aspect_value(base.aspect),
        disabled=_aspect_mode(base.aspect) != 'custom',
        width=118,
    )

    def _redraw():
        for entry in entries:
            entry['pane'].object = _make_fig(entry['sn'], entry['anno'], title)

    def _set_axes(y_axis: int, x_axis: int):
        if sync['axes']:
            return
        if y_axis == x_axis:
            candidates = [axis for axis in range(base.ndim) if axis != y_axis]
            x_axis = candidates[0]
        sync['axes'] = True
        _sync_axis_select_options(y_axis, x_axis)
        sync['axes'] = False
        for sn in sns:
            if y_axis < sn.ndim and x_axis < sn.ndim:
                sn.set_display_axes((y_axis, x_axis))
        _rebuild_hidden_controls()
        _redraw()

    def _axes_changed(event):
        if sync['axes'] or sync['axis_options']:
            return
        _set_axes(y_select.value, x_select.value)

    def _sync_axis_select_options(y_axis: int, x_axis: int):
        sync['axis_options'] = True
        y_select.options = _axis_options(base, exclude=x_axis)
        x_select.options = _axis_options(base, exclude=y_axis)
        y_select.value = y_axis
        x_select.value = x_axis
        sync['axis_options'] = False

    def _rebuild_hidden_controls():
        children = []
        index_widgets = []
        hidden_axes = list(base.hidden_axes)
        if not hidden_axes:
            children.append(_muted('No fixed dimensions'))
        for axis in hidden_axes:
            max_idx = min(sn.shape[axis] for sn in sns if axis < sn.ndim) - 1
            value = min(base.get_index(axis), max_idx)
            idx_input = pn.widgets.IntInput(
                name=base.axis_name(axis),
                start=0,
                end=max_idx,
                value=value,
                width=118,
            )

            def _make_cb(axis=axis, idx_input=idx_input):
                def _cb(event=None):
                    for sn in sns:
                        if axis < sn.ndim and axis not in sn.display_axes:
                            sn.set_index(axis, idx_input.value)
                    _redraw()
                return _cb

            idx_input.param.watch(_make_cb(), 'value')
            index_widgets.append(idx_input)
        children.extend(_compact_rows(index_widgets))
        hidden_controls[:] = children

    def _update_clim(event=None):
        if sync['clim']:
            return
        clim = (float(vmin_w.value), float(vmax_w.value))
        for sn in sns:
            sn.clim = clim
        _redraw()

    def _auto_clim(event=None):
        vmin, vmax = _percentile_clim_many([sn.get_frame() for sn in sns])
        sync['clim'] = True
        vmin_w.value = _nice_float(vmin)
        vmax_w.value = _nice_float(vmax)
        sync['clim'] = False
        for sn in sns:
            sn.clim = (vmin, vmax)
        _redraw()

    def _update_cmap(event=None):
        from cigvis import colormap as cmap_mod
        cmap = cmap_mod.get_cmap_from_str(cmap_sel.value)
        for sn in sns:
            sn.cmap = cmap
        _redraw()

    def _update_aspect(event=None):
        if sync['aspect']:
            return
        if aspect_mode.value == 'auto':
            aspect = 'auto'
            sync['aspect'] = True
            aspect_value.disabled = True
            aspect_value.value = 1.0
            sync['aspect'] = False
        elif aspect_mode.value == 'equal':
            aspect = 1.0
            sync['aspect'] = True
            aspect_value.disabled = True
            aspect_value.value = 1.0
            sync['aspect'] = False
        else:
            aspect_value.disabled = False
            aspect = max(float(aspect_value.value), 1e-9)
        for sn in sns:
            sn.aspect = aspect
        _redraw()

    y_select.param.watch(_axes_changed, 'value')
    x_select.param.watch(_axes_changed, 'value')
    vmin_w.param.watch(_update_clim, 'value')
    vmax_w.param.watch(_update_clim, 'value')
    auto_btn.on_click(_auto_clim)
    cmap_sel.param.watch(_update_cmap, 'value')
    aspect_mode.param.watch(_update_aspect, 'value')
    aspect_value.param.watch(_update_aspect, 'value')
    _rebuild_hidden_controls()

    return _make_sidebar(
        title,
        subtitle,
        _section('View dimensions',
                 pn.Row(y_select, x_select, width=248)),
        _section('Fixed indices',
                 hidden_controls),
        _section('Aspect',
                 pn.Row(aspect_mode, aspect_value, width=248)),
        _section('Display',
                 pn.Row(vmin_w, vmax_w, width=248),
                 auto_btn,
                 cmap_sel),
    )


def _make_plot_pane(sn: SliceNode, anno_traces: List,
                    title: str) -> pn.pane.Plotly:
    return pn.pane.Plotly(
        _make_fig(sn, anno_traces, title),
        config={'scrollZoom': False, 'responsive': True},
        height=430,
        sizing_mode='stretch_width',
    )


def _make_sidebar(title: str, subtitle: str, *sections):
    display_title = html.escape(title or 'SliceViewer')
    display_subtitle = html.escape(subtitle)
    header = pn.pane.HTML(
        f'<div class="sliceviewer-title">{display_title}</div>'
        f'<div class="sliceviewer-subtitle">{display_subtitle}</div>',
        width=248,
    )
    return pn.Column(
        header,
        *sections,
        css_classes=['sliceviewer-sidebar'],
        width=284,
        min_width=284,
    )


def _section(title: str, *children):
    label = pn.pane.HTML(
        f'<div class="sliceviewer-section-title">{html.escape(title)}</div>',
        width=248,
    )
    return pn.Column(
        label,
        *children,
        css_classes=['sliceviewer-section'],
        width=248,
    )


def _muted(text: str):
    return pn.pane.HTML(
        f'<div class="sliceviewer-fixed-range">{html.escape(text)}</div>',
        width=248,
    )


def _compact_rows(widgets: List) -> List:
    rows = []
    for start in range(0, len(widgets), 2):
        rows.append(pn.Row(*widgets[start:start + 2], width=248))
    return rows


def _make_app_layout(controls, panes: List[pn.pane.Plotly],
                     grid: Optional[Tuple[int, int]]):
    rows, ncols = _grid_shape(len(panes), grid)
    plot_cards = [
        pn.Column(
            pane,
            css_classes=['sliceviewer-plot-card'],
            sizing_mode='stretch_width',
        )
        for pane in panes
    ]
    if grid is not None:
        plot_cards.extend(
            pn.Spacer(height=446, sizing_mode='stretch_width')
            for _ in range(rows * ncols - len(plot_cards))
        )
    content = pn.GridBox(*plot_cards, ncols=ncols, sizing_mode='stretch_width')
    return pn.Row(
        controls,
        content,
        css_classes=['sliceviewer-app'],
        sizing_mode='stretch_width',
    )


def _grid_shape(count: int, grid: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if grid is not None:
        if len(grid) != 2:
            raise ValueError("grid must be (rows, cols)")
        rows, cols = int(grid[0]), int(grid[1])
        if rows <= 0 or cols <= 0:
            raise ValueError("grid rows and cols must be positive")
        if rows * cols < count:
            raise ValueError(
                f"grid={grid!r} has {rows * cols} cells, but {count} panels "
                "were provided"
            )
        return rows, cols
    if count <= 3:
        return 1, max(1, count)
    cols = int(math.ceil(math.sqrt(count)))
    return int(math.ceil(count / cols)), cols


def _load_extension():
    pn.extension('plotly', raw_css=[_RAW_CSS])


def _ws_origins(address: str, port: int):
    if address in ('0.0.0.0', '::'):
        return '*'
    origins = {f'{address}:{port}', f'localhost:{port}', f'127.0.0.1:{port}'}
    return sorted(origins)


def _get_slice_node(nodes: List) -> Optional[SliceNode]:
    return next((n for n in nodes if isinstance(n, SliceNode)), None)


def _make_fig(sn: SliceNode, anno_traces: List, title: str) -> go.Figure:
    rgba = sn.render()
    yaxis = dict(
        autorange='reversed',
        title=sn.axis_short_name(sn.y_axis),
        showgrid=False,
        zeroline=False,
    )
    if sn.aspect != 'auto':
        yaxis.update(scaleanchor='x', scaleratio=float(sn.aspect))

    return go.Figure(
        data=[go.Image(z=rgba, name='slice')] + list(anno_traces),
        layout=go.Layout(
            template='plotly_white',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(
                family='Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',
                color='#334155',
                size=13,
            ),
            margin=dict(l=44, r=10, t=12, b=34),
            uirevision='keep',
            hovermode=False,
            xaxis=dict(
                constrain='domain',
                title=sn.axis_short_name(sn.x_axis),
                showgrid=False,
                zeroline=False,
            ),
            yaxis=yaxis,
            height=410,
        ),
    )


def _axis_options(sn: SliceNode, exclude: Optional[int] = None):
    return {
        sn.axis_name(axis): axis
        for axis in range(sn.ndim)
        if axis != exclude
    }


def _aspect_mode(aspect: Union[str, float]) -> str:
    if aspect == 'auto':
        return 'auto'
    if float(aspect) == 1.0:
        return 'equal'
    return 'custom'


def _aspect_value(aspect: Union[str, float]) -> float:
    return 1.0 if aspect == 'auto' else _nice_float(aspect)


def _nice_float(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        return 0.0
    return float(f'{value:.6g}')


def _percentile_clim_many(frames: List[np.ndarray], p_low=2, p_high=98):
    finite = [np.asarray(frame)[np.isfinite(frame)] for frame in frames]
    finite = [arr.ravel() for arr in finite if arr.size]
    if not finite:
        return 0.0, 1.0
    data = np.concatenate(finite)
    vmin = float(np.percentile(data, p_low))
    vmax = float(np.percentile(data, p_high))
    if vmin == vmax:
        pad = abs(vmin) * 0.01 or 1.0
        return vmin - pad, vmax + pad
    return vmin, vmax


def _cmap_name(cmap) -> str:
    if isinstance(cmap, str):
        return cmap if cmap in _CMAPS else _CMAPS[0]
    return getattr(cmap, 'name', _CMAPS[0]) if getattr(cmap, 'name', '') in _CMAPS else _CMAPS[0]
