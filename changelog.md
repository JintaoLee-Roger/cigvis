# Changelog


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