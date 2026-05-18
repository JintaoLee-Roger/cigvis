# Changelog


### v0.3.1

**Fixed**

- Fixed transparent-background PNG export so VisPy colorbar visuals keep their
  alpha channel and remain visible in saved images.

**Changed**

- Updated package metadata to require Python >= 3.9.


### v0.3.0

This release reorganizes the public plotting APIs, replaces the old desktop GUI,
and separates backend-specific interfaces more explicitly.

**Added**

- Added structured `plot3D` option objects:
  - `cigvis.Plot3DView` for canvas, layout, and camera settings.
  - `cigvis.Plot3DSave` for automatic screenshots.
  - `cigvis.Plot3DColorbar` for colorbar export settings.
  - `cigvis.Plot3DGui` for the optional PySide6 GUI shell.
- Added automatic screenshot export through `Plot3DSave`; PNG export uses the current canvas framebuffer and supports transparent backgrounds.
- Added `plot3D(..., gui=True)` integration with the modern 3D GUI shell, so existing `plot3D` nodes can be opened with a lightweight control panel.
- Added axis-aware slice sources for VisPy, Viser, and Plotly slice APIs, so `volume` may be a dict such as `{'x': iline_source, 'y': xline_source, 'z': time_source}`.
- Added per-source physical axis order declarations such as `{'data': time_source, 'axes': ('z', 'y', 'x')}`, allowing each slice direction to use a source optimized for its own storage layout.
- Added `display_range` for slice APIs to limit visible x/y/z ranges and push those ranges into slice reads, which is useful for large lazy volumes with unused samples such as deep time/depth tails.
- Added camera-relative `HeadlightShadingFilter` for mesh, surface, point, and well-log lighting. Lighting is now attached to the visual node itself rather than globally managed by `VisCanvas`.
- Added Gaussian splat rendering (`Splat`) for dense point/voxel-style displays.
- Added `create_surfaces(..., quad=True, quad_size=...)` support for point-style horizon patch rendering, where each `(x, y, z)` point becomes an independent quad patch.
- Added editable `line_cmap` support in GUI colormap controls for 2D images/masks and 3D base slices, masks, and surfaces. GUI expressions such as `line_cmap`, `line_cmap('jet')`, and `line_cmap(28, 256)` are supported, with default samples inferred from the volume z length when available.
- Added a Panel+Plotly `sliceviewer` backend for SSH/Jupyter-friendly 2D viewing of 2D/3D/4D arrays, including runtime dimension selection and automatic two-largest-dimension defaults for thin-volume workflows.

**Changed**

- Replaced the old PyQt5 GUI implementation with PySide6-backed GUI components. The importable `cigvis.gui` package remains as a compatibility layer, while the retained GUI surface is the `plot3D(gui=True)` inspector.
- Removed the standalone `cigvis.gui.gui2d()` and `cigvis.gui.gui3d()` viewers from `cigvis`. `plot3D(gui=True)` remains available as a lightweight plot inspector for existing VisPy nodes.
- Simplified the 3D GUI sidebar and removed GUI-level Shading/Lighting controls. Mesh, surface, point, and well-log lighting should now be configured when creating nodes, for example through `create_surfaces(..., shading=..., dyn_light=...)`.
- Fixed surface interpolation handling and made point-style quad surfaces handle depth coloring and valid `z=0` points correctly.
- Removed the old `VisCanvas` light-management mixin; camera-relative lighting is now handled by `HeadlightShadingFilter` on each visual.
- `plot3D` no longer uses `dyn_light` as a canvas-level option; node lighting is controlled at node creation time.
- `cigvis` no longer switches its top-level `create_*` functions to Plotly implementations automatically inside Jupyter notebooks. The top-level `cigvis.create_*` functions are the VisPy API when VisPy is available.
- Plotly functions are now backend-specific: use `cigvis.plotlyplot.create_*` explicitly for Plotly workflows. Do not rely on `cigvis.create_*` in notebooks to mean Plotly.
- Renamed Plotly helpers to snake_case, including `plotlyplot.create_bodies` and `plotlyplot.create_line_logs`.
- `vispy` is now a core dependency; desktop GUI dependencies moved from PyQt5 to PySide6 through the `gui` extra.
- Added the `sliceviewer` optional dependency extra.
- Added `sliceviewer.build_layout(...)` and `sliceviewer.show(..., launch=False)` for tests or embedding without starting a Panel server.
- Improved `sliceviewer` defaults and layout: local serving now binds to `localhost` by default, and the control panel uses a compact high-contrast sidebar with display-axis, fixed-index, aspect, and comparison-grid controls.
- Cleaned stale kwarg filtering helpers from `cigvis.utils.vispyutils`; new user-facing `plot3D` options are exposed through the `Plot3D*` dataclasses instead.
- Cleaned colormap helpers by removing old deprecated helpers/parameters such as `blend_two_arrays`, `blend_multiple`, `includevispy`, and `forvispy`.
- Renamed `AxisAlignedImage.set_visable` to `set_visible`.
- Fixed several small Viser/volume-slice issues, including linked-line updates when some axes are absent and the "parameters" GUI label typo.
- Extended the retained `plot3D(gui=True)` GUI shell and internal GUI components with display-range controls, splat loading/management, overlay removal cleanup, and non-blocking file loading with busy feedback.
- Added display-range Apply/Reload controls and made splat layer items appear only after the splat node is created successfully.
- Updated examples, README, and docs to use the new `plot3D(view=..., save=..., cbar=..., gui=...)` API style.
- Updated transparent-background documentation to use `Plot3DSave(transparent_bg=True)` instead of post-processing a special background color.
- PNG screenshot/export now uses transparent backgrounds by default; pass `Plot3DSave(transparent_bg=False)` for a solid background.
- Removed the independent `Plot3DSave.size`/`output_policy` export-size controls. Use `Plot3DView(size=...)` to control the canvas size; automatic saves and the `s` shortcut now share the same framebuffer capture path.
- `Plot3DSave` screenshot options are also passed to the `s` shortcut unless `Plot3DView(shortcut_save_kw=...)` is set explicitly.
- PNG screenshot/export now captures the visible framebuffer directly and no longer re-composites colorbars after capture, avoiding duplicate colorbar labels in saved images.
- Colorbars created by `create_colorbar_from_nodes` now carry source metadata, and `plot3D(gui=True)` keeps matching slice, mask, and surface colorbars in sync when GUI colormap/range controls change.
- Colorbar image rendering now uses Matplotlib's Agg canvas directly, avoiding pyplot/Qt backend work during GUI callbacks.
- Surface/point splats now write depth by default to avoid dense translucent layers smearing over each other during camera or slice movement; volume splats keep the translucent-cloud behavior.
- `line_cmap` can now be called as `line_cmap(28, 256)` to mean `n_lines=28, samples=256`; mask alpha controls scale only the opaque line entries and keep transparent background entries transparent.
- `auto_clim` now samples non in-memory array-like inputs instead of forcing a full array read.
- Updated user-facing comments and examples to use English comments consistently.

**Breaking / Migration Notes**

- In Jupyter notebooks, use `from cigvis import plotlyplot` and call `plotlyplot.create_*` / `plotlyplot.plot3D(...)` for Plotly output. Do not expect `cigvis.create_*` to dispatch to Plotly automatically.
- Plotly notebook code should use the Plotly namespace and snake_case names, e.g. `plotlyplot.create_bodies` and `plotlyplot.create_line_logs`.
- For VisPy rendering, keep using `cigvis.create_*` and `cigvis.plot3D(...)`.
- For remote/browser 3D visualization, use `cigvis.viserplot`.
- For lightweight remote/browser 2D slice viewing, use `cigvis.sliceviewer`.

**Deprecated**

- Deprecated legacy top-level `plot3D` parameters such as `size=`, `savename=`, `grid=`, `share=`, `xyz_axis=`, and `cbar_region_ratio=`. These are deprecated since `0.2.1` and scheduled for removal in `0.4.0`; use `view=...`, `save=...`, `cbar=...`, and `gui=...` instead.
- Deprecated `plot3D(dyn_light=...)`. Pass `dyn_light` to node creation functions such as `create_surfaces`, `create_bodies`, `create_points`, or `create_fault_skin` instead. This compatibility path is scheduled for removal in `0.4.0`.
- Deprecated spelling/legacy aliases remain available in the top-level VisPy API with warnings until `0.4.0`, including `create_bodys` and `create_Line_logs`; use `create_bodies` and `create_line_logs`.
- Deprecated the old surface input style where value/color arrays are combined directly into `surfs` with `ndim > 2`. Put values or color matrices in `value_type` instead.


### v0.2.0

- fixed `stratum` colormap.


### v0.1.9

- added intersection lines to make slices easier to distinguish
- added a function to link multiple servers, so that we can synchronize multiple canvases (in different tabs).
- the changed can be refered from this example [cigvis/gallery/viser/04](https://cigvis.readthedocs.io/en/latest/gallery/viser/04_comparison.html#sphx-glr-gallery-viser-04-comparison-py).


### v0.1.8

- added support for RGB/RGBA format 3D volumes as input
- improved viser-based experience by enhancing GUI panel, including:
    - added mask parameter controls
    - added region selection for screenshots
    - added ability to compare multiple results
- updated add_masks function to support quick colormap setup via `alpha` and `excpt` parameters
- separated installation dependencies: `vispy` and `PyQt5` are no longer installed with `cigvis[plotly]` or `cigvis[viser]`
- fixed several bugs


### v0.1.7

- added function `create_well_logs` for viser backend
- added intersection lines and border lines for volume slices


### v0.1.6

- improved documents and add blogs


### v0.1.5

- added `shading` and `dyn_light` to control the mesh's shadingfilter and dyanmic lightning


### v0.1.4

- fixed a bug reported by #20 and #21
- added mplstpyle file
- improved viserplot


### v0.1.3

- add `parula` and `batlow` colormap
- higher resolution colorbar
- fix a bug when the alpha is not 1
- support to print states in viser
- gui support high dpi scaling

### v0.1.2

- added `blue_white_purple` colormap
- added `nancolor` to set nan color in viser
- added `extract_arbitrary_line_by_view` function to a extract arbitrary line by clicking


### v0.1.1

- provided a higher resolution colorbar image
- formatted axis tick labels, e.g., "0.85000001" to "0.85"
- improved the deprecated information.


### v0.1.0

**Breaking Changes**

- added `viserplot` for remote display 3D volume in web (based on `viser`)
- added `Axis3d` to create axis
- added `SurfaceNode` and `ArbLineNode` to create surface and arbitrary lines
- Improved a number of interface optimizations, including the creation of colorbars, the overlay display of faces, whether or not to change the light, and so on.
- improved `gui3d`, support import surface
- added extra dependencies option to selectively install dependencies
- dealed with `nan`
- support SEG-Y file (powered by `cigsegy`)
- ...
