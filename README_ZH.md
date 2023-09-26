# CIGVis - a tool for geophysical data visualization

## 概述

**cigvis** 是一个由[Computational Interpretation Group (CIG)](https://cig.ustc.edu.cn/main.htm)开发的用于可视化多维地球物理数据的工具，只需几行代码即可快速可视化数据。

cigvis可用于各种地球物理数据的可视化，包括3D地震数据、与标签数据叠加、断层、RGT、层位、测井轨迹和测井曲线等其他信息，3D地质体、2D数据和1D数据等等。其GitHub仓库位于[github.com/JintaoLee-Roger/cigvis](https://github.com/JintaoLee-Roger/cigvis)，文档可在[https://cigvis.readthedocs.io/](https://cigvis.readthedocs.io/)找到。

cigvis利用底层库的强大功能，例如[vispy](https://github.com/vispy/vispy)用于3D可视化，[matplotlib](https://matplotlib.org/)用于2D和1D可视化，以及[plotly](https://plotly.com/)用于Jupyter环境（正在开发中）。3D可视化组件在[yunzhishi/seismic-canvas](https://github.com/yunzhishi/seismic-canvas)的代码基础上进行了大量开发。

## 安装

通过PyPI安装，请使用以下命令：
```shell
pip install cigvis
```

要进行本地安装，请从GitHub克隆存储库，然后使用pip进行安装：
```shell
git clone https://github.com/JintaoLee-Roger/cigvis.git
pip install -e .
```

## 核心特点

1. 方便的3D地球物理数据可视化。
2. 持续开发2D和1D数据可视化。
3. 针对地球物理数据的附加颜色映射。
4. 在OpenVDS格式中快速显示大数据。

## 使用

### 基本结构

cigvis的可视化代码的基本结构包括：
1. 数据加载
2. 创建节点
3. 将节点传递给`plot3D`函数

例如：
```python
import numpy as np
import cigvis

# 加载数据
d = np.fromfile('sx.dat', np.float32).reshape(ni, nt, nx)

# 创建节点
nodes = cigvis.create_slices(d)

# 在3D中可视化
cigvis.plot3D(nodes)
```

这个基本代码结构允许您使用cigvis快速可视化地球物理数据。只需加载数据，创建节点，并将它们传递给`plot3D`函数，就像上面的示例中所演示的那样。

### 摄像机和拖动

左键单击并拖动以旋转摄像机角度；右键单击并拖动，或滚动鼠标滚轮，进行缩放。按住`<Shift>`键，左键单击并拖动以平移移动。按`<Space>`键返回初始视图。按`<S>`键随时保存屏幕截图PNG文件。按`<Esc>`键关闭窗口。

按住`<Ctrl>`键，在鼠标悬停在可选择的可视节点上时，可视节点将被突出显示；左键单击并拖动以移动突出显示的可视节点。体积切片将在拖动期间实时更新其内容。还可以按`<D>`键切换拖动模式的开/关。

按`<z>`键缩放z轴，按`<Z>`键或`<Shift> + <z>`键缩放z轴。按`<f>`键增加`fov`值，按`<F>`键或`<Shift> + <f>`键减小`fov`值。

![ex1](https://raw.githubusercontent.com/JintaoLee-Roger/images/main/cigvis/ex.gif)

按`<a>`键打印摄像机的参数，并按`<s>`键保存屏幕截图。

### 各种地球物理数据

在cigvis中，我们将各种地球物理数据表示为独立的节点，将这些节点组装成列表，然后将此列表传递给`plot3D`函数进行可视化。

我们将三维数据体可视化为沿x、y和z方向的多个切片。此外，我们可以在这些切片上叠加其他三维数据切片，允许用户使用鼠标沿一个轴交互拖动它们。

层位数据可以表示为（N，3）形状的散点，或者表示为尺寸为（n1，n2）的规则网格上的z值。

测井轨迹显示为管道，其中第一个测井曲线的尺寸由沿管道的每个位置的颜色和半径表示。其他测井曲线显示为附加到管道边缘的表面。下面显示了一个示例（代码可在[cigvis/gallery/3Dvispy/09](https://cigvis.readthedocs.io/en/latest/gallery/3Dvispy/09-slice_surf_body_logs.html#sphx-glr-gallery-3dvispy-09-slice-surf-body-logs-py)找到）。

![09](https://raw.githubusercontent.com/JintaoLee-Roger/images/main/cigvis/3Dvispy/09.png)

cigvis内部的这些功能允许以多种方式交互可视化各种地球物理数据类型，增强了在地球科学应用中对此类数据的理解和分析。

### 一个画布上的多个体积数据

在指定网格（例如`grid=(2,2)`）的同时，您可以将多个独立的节点组合传
递给`plot3D`函数。这允许您将画布分成多个独立的子画布，每个子画布在同一画布中显示不同的3D数据集。有关此功能的示例代码可以在文档中找到[cigvis/gallery/3Dvispy/10](https://cigvis.readthedocs.io/en/latest/gallery/3Dvispy/10-multi_canvas.html#sphx-glr-gallery-3dvispy-10-multi-canvas-py)。

![10](https://raw.githubusercontent.com/JintaoLee-Roger/images/main/cigvis/3Dvispy/10.gif)

此外，您可以将所有子画布的摄像机链接在一起（只需将`share=True`传递给`plot3D`函数）。这意味着在一个子画布中进行的旋转、缩放或切片操作将同时在所有其他子画布中进行镜像，确保它们同时展示相同的更改。这个功能在比较多组数据时非常有优势，比如不同实验的结果、与标签一起的结果、地震数据与属性的结果等等。
您可以在文档中找到此功能的示例代码[cigvis/gallery/3Dvispy/11](https://cigvis.readthedocs.io/en/latest/gallery/3Dvispy/11-share_cameras.html#sphx-glr-gallery-3dvispy-11-share-cameras-py)。

![11](https://raw.githubusercontent.com/JintaoLee-Roger/images/main/cigvis/3Dvispy/11.gif)

这些功能为使用cigvis在一个单一画布内强大地可视化和比较多个独立的3D数据集提供了有效的方式。


## 示例所用的数据

所有示例所用到的数据可以在 [https://rec.ustc.edu.cn/share/19a16120-5c42-11ee-a0d4-4329aa6b754b](https://rec.ustc.edu.cn/share/19a16120-5c42-11ee-a0d4-4329aa6b754b) 下载，密码：`1234`

## 示例库

请参阅：[cigvis/gallery](https://cigvis.readthedocs.io/gallery)