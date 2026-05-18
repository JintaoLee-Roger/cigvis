![im1](https://raw.githubusercontent.com/JintaoLee-Roger/images/main/cigvis/more_demos/070.png)

# CIGVis - 多维地球物理数据可视化工具

## 概述

**cigvis** 是由 [Computational Interpretation Group (CIG)](https://cig.ustc.edu.cn/main.htm) 开发的多维地球物理数据可视化工具。用户只需要几行代码，就可以快速浏览和交互分析数据。

cigvis 可用于多类地球物理数据可视化，包括 3D 地震数据、标签/断层/RGT 等体数据叠加、层位曲面、测井轨迹和测井曲线、3D 地质体、2D 数据和 1D 数据等。项目仓库位于 [github.com/JintaoLee-Roger/cigvis](https://github.com/JintaoLee-Roger/cigvis)，文档位于 [https://cigvis.readthedocs.io/](https://cigvis.readthedocs.io/)。

cigvis 使用 [vispy](https://github.com/vispy/vispy) 做桌面 3D 可视化，使用 [matplotlib](https://matplotlib.org/) 做 2D/1D 可视化，使用 [plotly](https://plotly.com/) 支持 Jupyter 环境，并使用 [viser](https://github.com/nerfstudio-project/viser) 支持适合 SSH/浏览器访问的 3D 可视化。3D 可视化组件基于 [yunzhishi/seismic-canvas](https://github.com/yunzhishi/seismic-canvas) 继续开发。

**CIGVis: An open-source Python tool for the real-time interactive visualization of multidimensional geophysical data**<br>
Jintao Li, Yunzhi Shi, Xinming Wu<br>
Paper: [https://library.seg.org/doi/abs/10.1190/geo2024-0041.1](https://library.seg.org/doi/abs/10.1190/geo2024-0041.1)

## 安装

通过 PyPI 安装：

```shell
# 桌面 3D，可使用 VisPy 和 PySide6
pip install "cigvis[gui]"

# Jupyter 中使用 Plotly
pip install "cigvis[plotly]"

# 浏览器/远程服务器中使用 Viser
pip install "cigvis[viser]"

# SSH 友好的 2D SliceViewer，基于 Panel + Plotly
pip install "cigvis[sliceviewer]"

# 安装全部可选依赖
pip install "cigvis[all]"
```

本地开发安装：

```shell
git clone https://github.com/JintaoLee-Roger/cigvis.git

# 桌面 3D
pip install -e ".[gui]" --config-settings editable_mode=compat

# Jupyter Plotly
pip install -e ".[plotly]" --config-settings editable_mode=compat

# 浏览器/远程服务器 Viser
pip install -e ".[viser]" --config-settings editable_mode=compat

# SSH 友好的 2D SliceViewer
pip install -e ".[sliceviewer]" --config-settings editable_mode=compat

# 全部依赖
pip install -e ".[all]" --config-settings editable_mode=compat
```

## 核心特点

1. 便捷的 3D 地球物理数据可视化。
2. 持续开发中的 2D 和 1D 数据可视化。
3. 针对地球物理数据设计的附加 colormap。
4. 快速显示 OpenVDS 格式的大数据。

## 使用

### 基本结构

cigvis 的可视化代码通常包含四步：

1. 加载数据。
2. 创建可视化节点。
3. 将节点传给 `plot3D`。
4. 按需分组设置视图、保存、色标和 GUI 参数。

例如：

```python
import numpy as np
import cigvis

# 加载数据
d = np.fromfile('sx.dat', np.float32).reshape(ni, nt, nx)

# 创建节点
nodes = cigvis.create_slices(d)

# 3D 可视化
cigvis.plot3D(nodes)
```

新代码推荐把 `plot3D` 的参数按职责分组：

```python
cigvis.plot3D(
    nodes,
    view=cigvis.Plot3DView(
        size=(900, 700),
        grid=(1, 2),
        share=True,
        xyz_axis=False,
        azimuth=-65,
        elevation=22,
    ),
    save=cigvis.Plot3DSave(
        path='example.png',
        transparent_bg=True,
    ),
    gui=cigvis.Plot3DGui(enabled=False),
)
```

`view` 控制 VisPy 画布、布局和相机；`save` 控制自动截图；`cbar` 控制 colorbar 导出；`gui` 控制可选的 PySide6 GUI 外壳。旧的顶层参数，例如 `size=`、`savename=`、`grid=`、`share=`、`xyz_axis=` 和 `cbar_region_ratio=`，已从 `0.2.1` 开始弃用，并计划在 `0.4.0` 移除。

后端 API 现在是显式的。桌面 VisPy 渲染使用顶层 `cigvis.create_*` 和 `cigvis.plot3D`。在 Jupyter 中不要依赖 `cigvis.create_*` 自动切换到 Plotly，请显式导入后端命名空间：

```python
from cigvis import plotlyplot

nodes = plotlyplot.create_slices(d)
# Plotly overlay 和 VisPy 使用同样的流程：
# nodes = plotlyplot.add_mask(nodes, mask, cmap='jet', interpolation='nearest')
plotlyplot.plot3D(nodes)
```

### 摄像机和拖动

左键拖动旋转相机；右键拖动或滚轮缩放；按住 `<Shift>` 后左键拖动平移。按 `<Space>` 返回初始视角，按 `<S>` 保存截图，按 `<Esc>` 关闭窗口。

按住 `<Ctrl>` 时，鼠标悬停到可选择节点上会高亮显示；左键拖动高亮节点可以移动它。体数据切片会在拖动时实时更新。也可以按 `<D>` 开关拖动模式。

按 `<z>` 放大 z 轴，按 `<Z>` 或 `<Shift> + <z>` 缩小 z 轴。按 `<f>` 增大 `fov`，按 `<F>` 或 `<Shift> + <f>` 减小 `fov`。

![ex1](https://raw.githubusercontent.com/JintaoLee-Roger/images/main/cigvis/ex.gif)

按 `<a>` 可以实时打印相机参数；按住 `<alt>`（macOS 上为 `<option>`）并左键点击三维体中的位置，可以显示点击点坐标。

![ex2](https://raw.githubusercontent.com/JintaoLee-Roger/images/main/cigvis/ex2.gif)

### 各类地球物理数据

在 cigvis 中，各类地球物理数据会被表示成独立节点，再组装成列表传给 `plot3D`。

三维体数据通常展示为沿 x、y、z 方向的多个切片，也可以把其他三维数据叠加到这些切片上，并通过鼠标沿轴交互拖动。

层位数据既可以表示为形状为 `(N, 3)` 的散点，也可以表示为规则网格 `(n1, n2)` 上的 z 值。

测井轨迹可以显示为管状轨迹，第一个测井曲线的值会映射到管线每个位置的颜色和半径，其他测井曲线可显示为附着在管线边缘的曲面。示例见 [cigvis/gallery/3Dvispy/09](https://cigvis.readthedocs.io/en/latest/gallery/3Dvispy/09-slice_surf_body_logs.html#sphx-glr-gallery-3dvispy-09-slice-surf-body-logs-py)。

![09](https://raw.githubusercontent.com/JintaoLee-Roger/images/main/cigvis/3Dvispy/09.png)

### 一个画布中的多个体数据

可以把多个独立节点组合传给 `plot3D`，并通过 `view=cigvis.Plot3DView(grid=(2, 2))` 指定网格。这样一个画布会被分成多个独立子画布，每个子画布显示一个 3D 数据集。示例见 [cigvis/gallery/3Dvispy/10](https://cigvis.readthedocs.io/en/latest/gallery/3Dvispy/10-multi_canvas.html#sphx-glr-gallery-3dvispy-10-multi-canvas-py)。

![10](https://raw.githubusercontent.com/JintaoLee-Roger/images/main/cigvis/3Dvispy/10.gif)

通过 `view=cigvis.Plot3DView(share=True)` 可以联动所有子画布相机。在一个子画布中的旋转、缩放或切片操作会同步到其他子画布，适合比较不同实验结果、地震数据与属性、结果与标签等。示例见 [cigvis/gallery/3Dvispy/11](https://cigvis.readthedocs.io/en/latest/gallery/3Dvispy/11-share_cameras.html#sphx-glr-gallery-3dvispy-11-share-cameras-py)。

![11](https://raw.githubusercontent.com/JintaoLee-Roger/images/main/cigvis/3Dvispy/11.gif)

## 基于 Web 的 3D 可视化

cigvis 基于 [viser](https://github.com/nerfstudio-project/viser) 支持浏览器中的 3D 可视化，适合远程服务器和 SSH 场景。通常只需要把 `cigvis` 后端替换为 `viserplot`：

```diff
    import numpy as np
    import cigvis
+   from cigvis import viserplot

    d = np.fromfile('sx.dat', np.float32).reshape(ni, nt, nx)

-   nodes = cigvis.create_slices(d)
+   nodes = viserplot.create_slices(d)

-   cigvis.plot3D(nodes)
+   viserplot.plot3D(nodes)
```

在 Jupyter 环境中，建议维护一个固定 server，避免端口变化：

```diff
    import numpy as np
    import cigvis
+   from cigvis import viserplot
+   server = viserplot.create_server(8080)

    d = np.fromfile('sx.dat', np.float32).reshape(ni, nt, nx)

-   nodes = cigvis.create_slices(d)
+   nodes = viserplot.create_slices(d)

-   cigvis.plot3D(nodes)
+   viserplot.plot3D(nodes, server=server)
```

调用 `viserplot.plot3D` 后会输出服务地址。如果代码运行在本机，直接在浏览器打开 `0.0.0.0:8080` 即可；如果运行在远程服务器，可在浏览器中访问 `{ip}:8080`。

浏览器后端暂不支持在单个 tab 中划分多个 canvas。需要比较多个结果时，可以使用多个 server 作为替代方案，参考 [cigvis/gallery/viser/04](https://cigvis.readthedocs.io/en/latest/gallery/viser/04_comparison.html#sphx-glr-gallery-viser-04-comparison-py)。

![04](https://raw.githubusercontent.com/JintaoLee-Roger/images/main/cigvis/viser/04.gif)

更多示例见 [gallery/viser](https://cigvis.readthedocs.io/en/latest/gallery/viser/index.html)。

### SSH 友好的 2D SliceViewer

当远程服务器没有 OpenGL，或者 3D/4D 场景过重时，`cigvis.sliceviewer` 提供了轻量的浏览器 2D 查看器。它会用 NumPy/Plotly 渲染指定二维平面，并通过 Panel 提供服务：

```python
import numpy as np
from cigvis import sliceviewer as sv

volume = np.fromfile('sx.dat', np.float32).reshape(4, ni, nx)
nodes = sv.create_slice(
    volume,
    display_axes=(1, 2),  # 渲染为 (Y, X)
    indices={0: 2},       # 隐藏维度的固定索引
    aspect=1.0,
    cmap='gray',
    interpolation='nearest',
    render_mode='float',
)

# SSH 场景下先转发端口：
# ssh -L 5007:localhost:5007 user@server
sv.show(nodes, port=5007)
```

侧边栏可以切换 `Y axis` / `X axis`、交换 X/Y、调整隐藏维度索引、设置纵横比，选择 RGBA 图像或 float heatmap 渲染，并选择插值方式（`nearest`、`linear`、`best` 或 `auto`）。如果省略 `display_axes`，默认显示最大的两个维度。

前后对比时，可以传入两个或三个节点列表并指定网格：

```python
nodes_raw = sv.create_slice(raw, display_axes=(2, 3), indices={0: 1, 1: 2})
nodes_out = sv.create_slice(processed, display_axes=(2, 3), indices={0: 1, 1: 2})

sv.show([nodes_raw, nodes_out], grid=(1, 2), port=5007)
```

对于脚本和测试，如果只需要 Panel 对象，可以调用 `sv.show(nodes, launch=False)` 或 `sv.build_layout(nodes)`。

`sliceviewer` 默认绑定到 `localhost`，适合本地浏览和 SSH 端口转发。只有需要其他机器直接连接服务进程时，才使用 `address='0.0.0.0'`。

## 引用

如果 cigvis 对您的研究有帮助，请考虑引用：

Plain Text

```text
Li, J., Shi, Y. and Wu, X., 2024. CIGVis: an open-source python tool for real-time interactive visualization of multidimensional geophysical data. Geophysics, 90(1), pp.1-37.
```

BibTex

```bibtex
@article{li2024cigvis,
  title={CIGVis: an open-source python tool for real-time interactive visualization of multidimensional geophysical data},
  author={Li, Jintao and Shi, Yunzhi and Wu, Xinming},
  journal={Geophysics},
  volume={90},
  number={1},
  pages={1--37},
  year={2024},
  publisher={Society of Exploration Geophysicists}
}
```

## 示例数据

示例所用数据可以从 [https://rec.ustc.edu.cn/share/19a16120-5c42-11ee-a0d4-4329aa6b754b](https://rec.ustc.edu.cn/share/19a16120-5c42-11ee-a0d4-4329aa6b754b) 下载，密码：`1234`。

## 示例库

请参阅：[cigvis/gallery](https://cigvis.readthedocs.io/en/latest/gallery/index.html)
