# Changelog



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