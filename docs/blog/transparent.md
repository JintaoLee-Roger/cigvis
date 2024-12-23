如何获得背景透明的图像
=====================

当使用 cigvis 或类似的绘图库时，默认保存的图像背景颜色可能不是透明的。然而，通过以下几个步骤，我们可以将背景设置为透明：


## 1：使用超出colormap范围的背景色

为了确保背景颜色与绘制的数据区分开，选择一个不会出现在绘图所使用的颜色映射中的颜色。例如：

- 如果你使用的是 `jet` 颜色映射，避免使用蓝色、绿色或红色等颜色，这些都包含在渐变中。
- 一个好的选择可能是 `(0, 255, 0)`（纯绿色）或 `(255, 0, 255)`（品红色）。

下面是一个设置独特背景色的 cigvis 示例：

```python
import numpy as np
import cigvis
from cigvis import colormap

d = np.fromfile(xxxx).reshape(xxxx)
rgt = np.fromfile(xxxx).reshape(xxxx)

nodes = cigvis.create_slices(d, cmap='gray')
cmap = colormap.set_alpha('jet', 0.8)
nodes = cigvis.add_mask(nodes, rgt, cmaps=cmap)

cigvis.plot3D(nodes, bgcolor=(1, 0, 1, 1))  # 设置背景为品红色
```

## 2: 将背景颜色替换为透明

保存的 PNG 图像现在有一个自己设置的背景色。为了将其转换为透明背景，我们可以使用一个简单的 Python 脚本来替换背景颜色。以下是实现的代码：

```python
import sys
from PIL import Image
import numpy as np

def set_background_transparent(input_path, output_path, bg_color=(0, 255, 0)):
    """
    Replace the specified background color with transparency in a PNG image.

    :param input_path: Path to the input PNG file.
    :param output_path: Path to save the output PNG file with transparency.
    :param bg_color: The RGB color to be made transparent (default is green).
    """
    # Load the image and ensure it has an alpha channel
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img)

    # Extract RGBA channels
    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

    # Create a mask for the background color
    mask = (r == bg_color[0]) & (g == bg_color[1]) & (b == bg_color[2])

    # Set the alpha channel to 0 for the background pixels
    data[mask, 3] = 0

    # Save the modified image
    result = Image.fromarray(data, "RGBA")
    result.save(output_path)

# Main entry point for command-line usage
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python make_transparent.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    set_background_transparent(input_file, output_file)
```

运行脚本，将保存的 PNG 图像中的背景颜色替换为透明：
```bash
python make_transparent.py raw.png transparent.png
```

<!-- ![]() -->

## 推荐颜色


| Colormap        | 推荐背景颜色     |
|-----------------|------------------|
| gray            | (0, 0, 0)     |
| jet             | (255, 255, 255) |
| hot             | (0, 0, 255)  |
| cool            | (255, 0, 255)  |
| viridis         | (255, 255, 0) |
| plasma          | (0, 255, 0)   |



# English

When working with cigvis or similar plotting libraries, the default background color of a saved plot might not be transparent. However, we can make it transparent by leveraging a few steps:

1. Set the background to a unique color during plotting.
2.	Save the plot as a PNG file.
3.	Post-process the saved PNG to replace the unique background color with transparency.

This article walks you through these steps, with a Python script to automate the final step.

### Step 1: Use a Background Color Outside the Colormap Range

To ensure the background is distinguishable from your plotted data, choose a color that does not appear in the colormap used for the plot. For example:

- If you are using the jet colormap, avoid colors like blue, green, or red that are part of the gradient.
- A good choice might be (0, 255, 0) (pure green) or (255, 0, 255) (magenta).

Here is an example of setting a unique background color in cigvis:

```python
import numpy as np
import cigvis
from cigvis import colormap

d = np.fromfile(xxxx).reshape(xxxx)
rgt = np.fromfile(xxxx).reshape(xxxx)

nodes = cigvis.create_slices(d, cmap='gray')
cmap = colormap.set_alpha('jet', 0.8)
nodes = cigvis.add_mask(nodes, rgt, cmaps=cmap)

cigvis.plot3D(nodes, bgcolor=(1, 0, 1, 1))

# save
```

### Step 2: Replace the Background Color with Transparency

The saved PNG now has a solid background color. To make it transparent, we can replace the background color (e.g., green) with transparency using a simple Python script. Below is the implementation:

Python Script for Transparency

Save this script as make_transparent.py:

```python
import sys
from PIL import Image
import numpy as np

def set_background_transparent(input_path, output_path, bg_color=(0, 255, 0)):
    """
    Replace the specified background color with transparency in a PNG image.

    :param input_path: Path to the input PNG file.
    :param output_path: Path to save the output PNG file with transparency.
    :param bg_color: The RGB color to be made transparent (default is green).
    """
    # Load the image and ensure it has an alpha channel
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img)

    # Extract RGBA channels
    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

    # Create a mask for the background color
    mask = (r == bg_color[0]) & (g == bg_color[1]) & (b == bg_color[2])

    # Set the alpha channel to 0 for the background pixels
    data[mask, 3] = 0

    # Save the modified image
    result = Image.fromarray(data, "RGBA")
    result.save(output_path)

# Main entry point for command-line usage
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python make_transparent.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    set_background_transparent(input_file, output_file)
```

### Step 3: Automating the Transparency Process

Run the script on your saved PNG image to replace the background color with transparency:

```bash
python make_transparent.py plot_with_background.png plot_with_transparent_bg.png
```

- Input: plot_with_background.png (plot with a unique background color).
- Output: plot_with_transparent_bg.png (plot with a transparent background).

### Summary

1. Set a unique background color during plotting: Choose a color not in the colormap.
2.	Save the plot as a PNG file: Retain the unique background.
3.	Post-process to make the background transparent: Use the provided Python script to replace the background color with transparency.

With this approach, you can efficiently generate plots with transparent backgrounds, suitable for presentations, overlays, or further processing.
