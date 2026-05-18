import numpy as np

from vispy.color import ColorArray
from vispy.gloo import VertexBuffer
from vispy.visuals.shaders import Function, Variable
from vispy.visuals.visual import Visual
from vispy.util.event import Event
from vispy.scene.visuals import create_visual_node


__all__ = ["SplatVisual", "Splat"]


_SPLAT_VERTEX_SHADER = """
uniform float u_antialias;
uniform float u_px_scale;
uniform bool u_scaling;
uniform float u_canvas_size_min;
uniform float u_canvas_size_max;

attribute vec3 a_position;
attribute vec4 a_color;
attribute float a_size;

varying vec4 v_color;
varying float v_total_size;
varying float v_visible;

void main(void) {
    v_color = a_color;
    v_visible = 1.0;

    vec4 pos = vec4(a_position, 1.0);
    vec4 fb_pos = $visual_to_framebuffer(pos);

    vec4 size_vec;
    float v_size;

    if (u_scaling) {
        // "scene"/"visual" scaling: interpret a_size in scene/visual units
        pos = $framebuffer_to_scene_or_visual(fb_pos);
        float pos_w = pos.w == 0.0 ? 1.0 : pos.w;
        pos = vec4(pos.xyz / pos_w, 1.0);

        float fb_w = fb_pos.w == 0.0 ? 1.0 : fb_pos.w;
        vec2 fb_xy = fb_pos.xy / fb_w;

        size_vec = $scene_or_visual_to_framebuffer(pos + vec4(a_size, 0.0, 0.0, 0.0));
        float size_x_w = size_vec.w == 0.0 ? 1.0 : size_vec.w;
        float size_x = length(size_vec.xy / size_x_w - fb_xy);

        size_vec = $scene_or_visual_to_framebuffer(pos + vec4(0.0, a_size, 0.0, 0.0));
        float size_y_w = size_vec.w == 0.0 ? 1.0 : size_vec.w;
        float size_y = length(size_vec.xy / size_y_w - fb_xy);

        size_vec = $scene_or_visual_to_framebuffer(pos + vec4(0.0, 0.0, a_size, 0.0));
        float size_z_w = size_vec.w == 0.0 ? 1.0 : size_vec.w;
        float size_z = length(size_vec.xy / size_z_w - fb_xy);

        v_size = max(max(size_x, size_y), size_z);
    } else {
        // "fixed" scaling: a_size in pixels
        v_size = a_size * u_px_scale;
    }

    if (!(v_size > 0.0)) {
        v_visible = 0.0;
        v_size = 0.0;
    }
    v_size = min(v_size, 4096.0);

    // Optional canvas size clamping (in pixels)
    if (u_canvas_size_min >= 0.0) v_size = max(v_size, u_canvas_size_min);
    if (u_canvas_size_max >= 0.0) v_size = min(v_size, u_canvas_size_max);

    // Total size includes antialias ring (like Markers)
    float total_size = v_visible > 0.5 ? v_size + 4.0 * (1.5 * u_antialias) : 0.0;
    v_total_size = total_size;

    gl_Position = $framebuffer_to_render(fb_pos);
    gl_PointSize = max(total_size, 1.0);
}
"""


_SPLAT_FRAGMENT_SHADER = """
uniform float u_alpha;
uniform float u_antialias;
uniform float u_sigma_rel;   // sigma = u_sigma_rel * (core_radius)
uniform float u_cutoff;      // discard when gaussian < cutoff (e.g. 1e-3)
uniform bool  u_premultiply; // premultiply alpha for nicer blending

varying vec4 v_color;
varying float v_total_size;
varying float v_visible;

// A simple, stable gaussian splat in point sprite coordinates.
void main(void) {
    if (v_visible < 0.5) discard;

    // point coord in [0,1]
    vec2 pc = gl_PointCoord.xy;

    // map to [-1,1], radius r in [0, sqrt(2)]
    vec2 uv = pc * 2.0 - 1.0;
    float r2 = dot(uv, uv);

    // "core" radius is 1.0 in this normalized space (since uv in [-1,1])
    // sigma in this normalized space:
    float sigma = max(1e-4, u_sigma_rel);  // relative to uv-space radius
    float g = exp(-0.5 * r2 / (sigma * sigma));

    // optional smooth edge using antialias (very lightweight)
    // expand edge a bit; if you want harder/softer adjust sigma_rel and cutoff
    if (g < u_cutoff) discard;

    float a = v_color.a * u_alpha * g;

    if (u_premultiply) {
        gl_FragColor = vec4(v_color.rgb * a, a);
    } else {
        gl_FragColor = vec4(v_color.rgb, a);
    }
}
"""


class SplatVisual(Visual):
    """
    Gaussian splatting visual (GL_POINTS) for point clouds / voxel masks.

    Parameters
    ----------
    scaling : {"fixed","scene","visual"} or bool
        Same semantics as MarkersVisual.

        - "fixed": a_size in pixels
        - "scene": a_size in scene units (affected by camera zoom)
        - "visual": a_size in visual units (affected by Visual transform)

        Back-compat: False->"fixed", True->"scene"
    alpha : float
        Global opacity multiplier.
    antialias : float
        Antialiasing amount in pixels (only affects pointsize padding here).
    sigma_rel : float
        Gaussian sigma in normalized sprite space (uv in [-1,1]).
        Typical: 0.35 ~ 0.75. Bigger -> fatter/softer splat.
    cutoff : float
        Discard threshold on gaussian value (performance + crispness).
        Typical: 1e-3 ~ 1e-2.
    premultiply : bool
        Use premultiplied alpha output. Often looks nicer for dense splats.
    canvas_size_limits : tuple or None
        Optional point-size clamp in canvas pixels, ``(min, max)``.
    depth_test : bool
        Whether splats should be depth-tested against existing geometry.
    depth_mask : bool
        Whether splats write depth. Keep this enabled for dense surface-like
        splats so back layers do not keep blending over front layers. Disable
        it only for intentionally translucent volume clouds.
    """

    def __init__(self, scaling="fixed", alpha=1.0, antialias=1.0,
                 sigma_rel=0.55, cutoff=1e-2, premultiply=True,
                 canvas_size_limits=None, depth_test=True, depth_mask=True,
                 **kwargs):
        self._vbo = VertexBuffer()
        self._data = None
        self._scaling = "fixed"
        self._canvas_size_limits = None
        self._depth_test = bool(depth_test)
        self._depth_mask = bool(depth_mask)

        Visual.__init__(self, vcode=_SPLAT_VERTEX_SHADER, fcode=_SPLAT_FRAGMENT_SHADER)
        self._draw_mode = 'points'

        # canvas size clamp (optional)
        self.shared_program['u_canvas_size_min'] = -1.0
        self.shared_program['u_canvas_size_max'] = -1.0

        # splat controls
        self.alpha = alpha
        self.antialias = antialias
        self.sigma_rel = sigma_rel
        self.cutoff = cutoff
        self.premultiply = premultiply
        self.canvas_size_limits = canvas_size_limits

        self.events.add(data_updated=Event)

        if kwargs:
            self.set_data(**kwargs)

        self.scaling = scaling
        self.freeze()


    # -------------------
    # Public API
    # -------------------
    def set_data(self, pos=None, size=3.0, color='white'):
        """
        Parameters
        ----------
        pos : (N,2) or (N,3) array
            Marker positions.
        size : float or (N,) array
            Marker diameter in pixels (fixed) or in scene/visual units
            (scaling on).
        color : Color or (N,4) array
            Marker color.
        """
        if pos is None or len(pos) == 0:
            self._data = None
            self.events.data_updated()
            self.update()
            return

        pos = np.asarray(pos)
        if pos.ndim != 2 or pos.shape[1] not in (2, 3):
            raise ValueError("pos must have shape (N,2) or (N,3)")

        n = pos.shape[0]
        position = np.zeros((n, 3), dtype=np.float32)
        position[:, :pos.shape[1]] = pos.astype(np.float32, copy=False)

        size_arr = np.asarray(size, dtype=np.float32)
        if size_arr.ndim == 0:
            size_arr = np.full(n, size_arr, dtype=np.float32)
        elif size_arr.shape != (n,):
            raise ValueError("size must be scalar or shape (N,)")

        col = ColorArray(color).rgba.astype(np.float32)

        if col.ndim == 1:
            col = np.tile(col[None, :], (n, 1))
        elif col.ndim == 2 and col.shape[0] == 1 and col.shape[1] == 4:
            col = np.tile(col, (n, 1))
        elif col.shape != (n, 4):
            raise ValueError("color must be a single color (4,) / (1,4) or shape (N,4)")

        structured = np.zeros(n, dtype=[
            ('a_position', np.float32, 3),
            ('a_color',    np.float32, 4),
            ('a_size',     np.float32),
        ])
        structured['a_position'] = position
        structured['a_color'] = col
        structured['a_size'] = size_arr

        self._data = structured
        self._vbo.set_data(structured)

        self.shared_program['a_position'] = self._vbo['a_position']
        self.shared_program['a_color'] = self._vbo['a_color']
        self.shared_program['a_size'] = self._vbo['a_size']

        self.events.data_updated()
        self.update()

    def _apply_gl_state(self):
        blend_func = (
            ('one', 'one_minus_src_alpha')
            if self._premultiply
            else ('src_alpha', 'one_minus_src_alpha')
        )
        self.set_gl_state(depth_test=self._depth_test,
                          depth_mask=self._depth_mask,
                          blend=True,
                          blend_func=blend_func)

    # -------------------
    # Properties
    # -------------------
    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        scaling_modes = {
            False: "fixed",
            True: "scene",
            "fixed": "fixed",
            "scene": "scene",
            "visual": "visual",
        }
        if value not in scaling_modes:
            raise ValueError(f"Unknown scaling option: {value!r}")
        self._scaling = scaling_modes[value]
        self.shared_program['u_scaling'] = self._scaling != "fixed"
        self.update()

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, v):
        self._alpha = float(v)
        self.shared_program['u_alpha'] = self._alpha
        self.update()

    @property
    def antialias(self):
        return self._antialias

    @antialias.setter
    def antialias(self, v):
        self._antialias = float(v)
        self.shared_program['u_antialias'] = self._antialias
        self.update()

    @property
    def sigma_rel(self):
        return self._sigma_rel

    @sigma_rel.setter
    def sigma_rel(self, v):
        self._sigma_rel = float(v)
        self.shared_program['u_sigma_rel'] = self._sigma_rel
        self.update()

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, v):
        self._cutoff = float(v)
        self.shared_program['u_cutoff'] = self._cutoff
        self.update()

    @property
    def premultiply(self):
        return self._premultiply

    @premultiply.setter
    def premultiply(self, v):
        self._premultiply = bool(v)
        self.shared_program['u_premultiply'] = self._premultiply
        self._apply_gl_state()
        self.update()

    @property
    def depth_test(self):
        return self._depth_test

    @depth_test.setter
    def depth_test(self, value):
        self._depth_test = bool(value)
        self._apply_gl_state()
        self.update()

    @property
    def depth_mask(self):
        return self._depth_mask

    @depth_mask.setter
    def depth_mask(self, value):
        self._depth_mask = bool(value)
        self._apply_gl_state()
        self.update()

    @property
    def canvas_size_limits(self):
        return self._canvas_size_limits if hasattr(self, "_canvas_size_limits") else None

    @canvas_size_limits.setter
    def canvas_size_limits(self, value):
        if value is None:
            self._canvas_size_limits = None
            self.shared_program['u_canvas_size_min'] = -1.0
            self.shared_program['u_canvas_size_max'] = -1.0
            self.update()
            return
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("canvas_size_limits must be (min, max) or None")
        mn, mx = value
        self._canvas_size_limits = (mn, mx)
        self.shared_program['u_canvas_size_min'] = float(mn) if mn is not None else -1.0
        self.shared_program['u_canvas_size_max'] = float(mx) if mx is not None else -1.0
        self.update()

    # -------------------
    # VisPy hooks
    # -------------------
    def _prepare_transforms(self, view):
        view.view_program.vert['visual_to_framebuffer'] = view.get_transform('visual', 'framebuffer')
        view.view_program.vert['framebuffer_to_render'] = view.get_transform('framebuffer', 'render')

        scaling = self._scaling if self._scaling != "fixed" else "scene"
        view.view_program.vert['framebuffer_to_scene_or_visual'] = view.get_transform('framebuffer', scaling)
        view.view_program.vert['scene_or_visual_to_framebuffer'] = view.get_transform(scaling, 'framebuffer')

    def _prepare_draw(self, view):
        if self._data is None:
            return False
        view.view_program['u_px_scale'] = view.transforms.pixel_scale

    def _compute_bounds(self, axis, view):
        if self._data is None:
            return None
        pos = self._data['a_position']
        return (pos[:, axis].min(), pos[:, axis].max())


Splat = create_visual_node(SplatVisual)
Splat.__module__ = __name__
Splat.__doc__ = """Scene node wrapper for :class:`SplatVisual`."""
