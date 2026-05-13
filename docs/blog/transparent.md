如何保存透明背景图像
=====================

从当前接口开始，cigvis 可以在保存截图时直接生成透明背景 PNG，不再需要先设置一个特殊背景色，再用额外脚本把该颜色替换成透明。

## 直接保存透明背景

如果希望 `plot3D` 在创建画布后立刻保存一张透明背景图，使用 `Plot3DSave(transparent_bg=True)`：

```python
import numpy as np
import cigvis
from cigvis import colormap

d = np.fromfile("sx.dat", np.float32).reshape(ni, nx, nt)
rgt = np.fromfile("rgt.dat", np.float32).reshape(ni, nx, nt)

nodes = cigvis.create_slices(d, cmap="gray")
cmap = colormap.set_alpha("jet", 0.8)
nodes = cigvis.add_mask(nodes, rgt, cmaps=cmap)

cigvis.plot3D(
    nodes,
    view=cigvis.Plot3DView(size=(900, 700)),
    save=cigvis.Plot3DSave(
        path="transparent.png",
        transparent_bg=True,
    ),
)
```

导出图片尺寸来自当前可见 VisPy canvas 的实际 framebuffer。`Plot3DSave`
不再提供独立的 `size` 参数；如需改变基础画布尺寸，请调整
`Plot3DView(size=...)`。

## 交互后按 s 保存

交互窗口里按 `s` 保存的是调整视角后的当前画面。这个快捷键保存默认使用透明背景，不需要额外传入 `Plot3DSave(transparent_bg=True)`。`save=...` 只控制创建画布时是否自动保存一张图。

## 旧方法

旧版本常见做法是设置一个不会出现在 colormap 中的背景色，例如品红色或纯绿色，然后在保存后用 Pillow 把这个背景色替换为透明。这种方法容易误伤图像中真实存在的同色像素，因此新代码应优先使用 `transparent_bg=True`。


# English

How to Save Images with a Transparent Background
=================================================

cigvis can now save transparent-background PNG files directly. You no longer need to render with a special background color and replace that color in a post-processing step.

## Save Transparent PNG Directly

If you want `plot3D` to save a transparent-background image immediately after creating the canvas, use `Plot3DSave(transparent_bg=True)`:

```python
import numpy as np
import cigvis
from cigvis import colormap

d = np.fromfile("sx.dat", np.float32).reshape(ni, nx, nt)
rgt = np.fromfile("rgt.dat", np.float32).reshape(ni, nx, nt)

nodes = cigvis.create_slices(d, cmap="gray")
cmap = colormap.set_alpha("jet", 0.8)
nodes = cigvis.add_mask(nodes, rgt, cmaps=cmap)

cigvis.plot3D(
    nodes,
    view=cigvis.Plot3DView(size=(900, 700)),
    save=cigvis.Plot3DSave(
        path="transparent.png",
        transparent_bg=True,
    ),
)
```

The exported image size comes from the current visible VisPy canvas framebuffer.
`Plot3DSave` no longer provides a separate `size` option; adjust
`Plot3DView(size=...)` if you need a different base canvas size.

## Press s After Interaction

Pressing `s` in the interactive window saves the current adjusted view. This shortcut uses a transparent background by default; you do not need to pass `Plot3DSave(transparent_bg=True)`. The `save=...` option only controls the initial automatic export when the canvas is created.

## Legacy Approach

Older workflows often used a unique background color, such as magenta or green, and then replaced that color with transparency using Pillow. This can accidentally remove real pixels with the same color, so new code should prefer `transparent_bg=True`.
