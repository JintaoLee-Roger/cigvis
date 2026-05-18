import numpy as np
import numbers
from vispy.color import Color
from vispy.visuals.filters import Filter
from vispy.gloo import VertexBuffer
from vispy.visuals.shaders import Function


shading_vertex_template = """
varying vec3 v_normal_vec;
varying vec3 v_light_vec;
varying vec3 v_eye_vec;
varying vec4 v_pos_scene;

void prepare_shading() {
    // 1. Pass scene position to fragment shader
    vec4 pos_scene = $render2scene(gl_Position);
    v_pos_scene = pos_scene;

    // 2. Calculate Normal in scene coordinates
    vec4 normal_scene = $visual2scene(vec4($normal, 1.0));
    vec4 origin_scene = $visual2scene(vec4(0.0, 0.0, 0.0, 1.0));
    normal_scene /= normal_scene.w;
    origin_scene /= origin_scene.w;
    v_normal_vec = normalize(normal_scene.xyz - origin_scene.xyz);

    // 3. Calculate Eye Vector (View Direction)
    // This vector points from the camera eye to the vertex position.
    // We calculate it by transforming screen positions back to scene positions.
    vec4 pos_doc = $scene2doc(pos_scene);
    pos_doc /= pos_doc.w;
    vec4 pos_front = pos_doc;
    vec4 pos_back = pos_doc;
    // Create two points close to each other in depth to determine direction
    pos_front.z -= 1e-5;
    pos_back.z += 1e-5;
    pos_front = $doc2scene(pos_front);
    pos_back = $doc2scene(pos_back);
    // Vector from Eye -> Vertex
    v_eye_vec = normalize(pos_back.xyz / pos_back.w - pos_front.xyz / pos_front.w);

    // 4. Calculate Light Vector (Camera-Relative)
    // We define the light direction in Document (Screen) space so it follows the camera.
    // Then we transform it to Scene (World) space to interact with the mesh.
    
    // Point 1: The origin in screen space
    vec4 light_p1_doc = vec4(0.0, 0.0, 0.0, 1.0); 
    // Point 2: The tip of the light vector in screen space (controlled by user)
    vec4 light_p2_doc = vec4($light_dir, 1.0); 
    
    // Transform both points to scene space
    vec4 light_p1_scene = $doc2scene(light_p1_doc);
    vec4 light_p2_scene = $doc2scene(light_p2_doc);
    
    light_p1_scene /= light_p1_scene.w;
    light_p2_scene /= light_p2_scene.w;

    // The resulting vector is now in scene coords but locked to camera orientation
    v_light_vec = normalize(light_p2_scene.xyz - light_p1_scene.xyz);
}
"""


shading_fragment_template = """
varying vec3 v_normal_vec;
varying vec3 v_light_vec;
varying vec3 v_eye_vec;
varying vec4 v_pos_scene;

void shade() {
    if ($shading_enabled != 1) {
        return;
    }

    vec3 base_color = gl_FragColor.rgb;
    vec4 ambient_coeff = $ambient_coefficient;
    vec4 diffuse_coeff = $diffuse_coefficient;
    vec4 specular_coeff = $specular_coefficient;
    vec4 ambient_light = $ambient_light;
    vec4 diffuse_light = $diffuse_light;
    vec4 specular_light = $specular_light;
    float shininess = $shininess;

    // --- Normal Calculation ---
    vec3 normal = v_normal_vec;
    if ($flat_shading == 1) {
        // Flat shading: calculate normal using derivatives
        vec3 u = dFdx(v_pos_scene.xyz);
        vec3 v = dFdy(v_pos_scene.xyz);
        normal = cross(u, v);
    } else {
        // Smooth shading: use interpolated normal.
        // CRITICAL FOR LAYERS/MESHES:
        // Flip the normal if we are viewing the back face.
        // This ensures the back side is lit correctly.
        normal = gl_FrontFacing ? normal : -normal;
    }
    normal = normalize(normal);

    vec3 light_vec = normalize(v_light_vec);
    vec3 eye_vec = normalize(v_eye_vec);

    // --- Ambient Illumination ---
    vec3 ambient = ambient_coeff.rgb * ambient_coeff.a
                   * ambient_light.rgb * ambient_light.a;

    // --- Diffuse Illumination ---
    // Note: light_vec points FROM light source (camera relative) TO vertex.
    // We use -light_vec to get vector TO light.
    // Since normal is already flipped for back faces, this dot product works for both sides.
    float diffuse_factor = dot(-light_vec, normal);
    diffuse_factor = max(diffuse_factor, 0.0);
    
    vec3 diffuse = diffuse_factor
                   * diffuse_coeff.rgb * diffuse_coeff.a
                   * diffuse_light.rgb * diffuse_light.a;

    // --- Specular Illumination ---
    float specular_factor = 0.0;
    bool is_illuminated = diffuse_factor > 0.0;
    
    if (is_illuminated && shininess > 0.0) {
        // Blinn-Phong reflection
        // halfway vector between view direction (-eye_vec) and light direction (-light_vec)
        // Note: vispy's eye_vec usually points from eye to vertex, so -eye_vec points to eye.
        vec3 to_eye = -eye_vec;
        vec3 to_light = -light_vec;
        vec3 halfway = normalize(to_light + to_eye);
        
        float reflection_factor = clamp(dot(halfway, normal), 0.0, 1.0);
        specular_factor = pow(reflection_factor, shininess);
    }
    
    vec3 specular = specular_factor
                    * specular_coeff.rgb * specular_coeff.a
                    * specular_light.rgb * specular_light.a;

    // --- Final Color Combination ---
    vec3 color = base_color * (ambient + diffuse) + specular;
    gl_FragColor.rgb = color;
}
"""


def _as_rgba(intensity_or_color, default_rgb=(1.0, 1.0, 1.0)):
    if isinstance(intensity_or_color, numbers.Number):
        intensity = intensity_or_color
        return Color(default_rgb, alpha=intensity)
    color = intensity_or_color
    return Color(color)


class HeadlightShadingFilter(Filter):
    """
    Apply shading to a MeshVisual using a camera-relative light source
    (headlamp style).
    
    Key features:
    1. The light source follows the camera (camera-relative lighting).
    2. Supports screen-relative lighting offsets through light_dir.
    3. Supports double-sided lighting, useful for thin structures such as seismic horizons.
    
    Parameters
    ----------
    shading : str
        'flat' or 'smooth'.
    light_dir : tuple
        Camera/screen-relative light direction (x, y, z).
        (0, 0, 1) means the light points straight into the screen.
        (10, 5, 10) means the light comes from the upper-right screen direction.

    Other Parameters
    ----------------
    ``**kwargs``
        Other parameters match the original VisPy shading filter.
    """
    _shaders = {
        'vertex': shading_vertex_template,
        'fragment': shading_fragment_template,
    }

    def __init__(self, shading='flat',
                 ambient_coefficient=(1, 1, 1, 1),
                 diffuse_coefficient=(1, 1, 1, 1),
                 specular_coefficient=(1, 1, 1, 1),
                 shininess=100,
                 light_dir=(0, 0, 1),  # Point straight into the screen by default
                 ambient_light=(1, 1, 1, .25),
                 diffuse_light=(1, 1, 1, 0.7),
                 specular_light=(1, 1, 1, .25),
                 enabled=True):
        self._shading = shading

        self._ambient_coefficient = _as_rgba(ambient_coefficient)
        self._diffuse_coefficient = _as_rgba(diffuse_coefficient)
        self._specular_coefficient = _as_rgba(specular_coefficient)
        self._shininess = shininess

        # The light direction is camera-relative.
        self._light_dir = light_dir
        
        self._ambient_light = _as_rgba(ambient_light)
        self._diffuse_light = _as_rgba(diffuse_light)
        self._specular_light = _as_rgba(specular_light)

        self._enabled = enabled

        vfunc = Function(self._shaders['vertex'])
        ffunc = Function(self._shaders['fragment'])

        self._normals = VertexBuffer(np.zeros((0, 3), dtype=np.float32))
        self._normals_cache = None
        vfunc['normal'] = self._normals

        super().__init__(vcode=vfunc, fcode=ffunc)

    @property
    def enabled(self):
        """True to enable the filter, False to disable."""
        return self._enabled

    @enabled.setter
    def enabled(self, enabled):
        self._enabled = enabled
        self._update_data()

    @property
    def shading(self):
        """The shading method."""
        return self._shading

    @shading.setter
    def shading(self, shading):
        assert shading in (None, 'flat', 'smooth')
        self._shading = shading
        self._update_data()

    @property
    def light_dir(self):
        """
        The light direction relative to the camera (Document Space).
        Format: (x, y, z)
        Example: (0, 0, 1) points from eye to scene center.
        """
        return self._light_dir

    @light_dir.setter
    def light_dir(self, direction):
        direction = np.array(direction, float).ravel()
        if direction.size != 3 or not np.isfinite(direction).all():
            raise ValueError('Invalid direction %s' % direction)
        self._light_dir = tuple(direction)
        self._update_data()

    @property
    def ambient_light(self):
        return self._ambient_light

    @ambient_light.setter
    def ambient_light(self, light_color):
        self._ambient_light = _as_rgba(light_color)
        self._update_data()

    @property
    def diffuse_light(self):
        return self._diffuse_light

    @diffuse_light.setter
    def diffuse_light(self, light_color):
        self._diffuse_light = _as_rgba(light_color)
        self._update_data()

    @property
    def specular_light(self):
        return self._specular_light

    @specular_light.setter
    def specular_light(self, light_color):
        self._specular_light = _as_rgba(light_color)
        self._update_data()

    @property
    def ambient_coefficient(self):
        return self._ambient_coefficient

    @ambient_coefficient.setter
    def ambient_coefficient(self, color):
        self._ambient_coefficient = _as_rgba(color)
        self._update_data()

    @property
    def diffuse_coefficient(self):
        return self._diffuse_coefficient

    @diffuse_coefficient.setter
    def diffuse_coefficient(self, diffuse_coefficient):
        self._diffuse_coefficient = _as_rgba(diffuse_coefficient)
        self._update_data()

    @property
    def specular_coefficient(self):
        return self._specular_coefficient

    @specular_coefficient.setter
    def specular_coefficient(self, specular_coefficient):
        self._specular_coefficient = _as_rgba(specular_coefficient)
        self._update_data()

    @property
    def shininess(self):
        return self._shininess

    @shininess.setter
    def shininess(self, shininess):
        self._shininess = float(shininess)
        self._update_data()

    def _update_data(self):
        if not self._attached:
            return

        # Pass light_dir to the vertex shader as a screen-space vector.
        self.vshader['light_dir'] = self._light_dir

        self.fshader['ambient_light'] = self._ambient_light.rgba
        self.fshader['diffuse_light'] = self._diffuse_light.rgba
        self.fshader['specular_light'] = self._specular_light.rgba

        self.fshader['ambient_coefficient'] = self._ambient_coefficient.rgba
        self.fshader['diffuse_coefficient'] = self._diffuse_coefficient.rgba
        self.fshader['specular_coefficient'] = self._specular_coefficient.rgba
        self.fshader['shininess'] = self._shininess

        self.fshader['flat_shading'] = 1 if self._shading == 'flat' else 0
        self.fshader['shading_enabled'] = (
            1 if self._enabled and self._shading is not None else 0
        )

        normals = self._visual.mesh_data.get_vertex_normals(indexed='faces')
        if normals is not self._normals_cache:
            self._normals_cache = normals
            self._normals.set_data(self._normals_cache, convert=True)

    def on_mesh_data_updated(self, event):
        self._update_data()

    def _attach(self, visual):
        super()._attach(visual)

        # Fetch the coordinate transform matrices required by the shader.
        render2scene = visual.transforms.get_transform('render', 'scene')
        visual2scene = visual.transforms.get_transform('visual', 'scene')
        scene2doc = visual.transforms.get_transform('scene', 'document')
        doc2scene = visual.transforms.get_transform('document', 'scene')
        
        self.vshader['render2scene'] = render2scene
        self.vshader['visual2scene'] = visual2scene
        self.vshader['scene2doc'] = scene2doc
        self.vshader['doc2scene'] = doc2scene

        if self._visual.mesh_data is not None:
            self._update_data()

        visual.events.data_updated.connect(self.on_mesh_data_updated)

    def _detach(self, visual):
        visual.events.data_updated.disconnect(self.on_mesh_data_updated)
        super()._detach(visual)
