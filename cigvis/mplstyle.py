_spectrum = """
font.family: 'Arial', 'SimHei'
xtick.minor.visible: True
ytick.minor.visible: True
xtick.minor.size: 4
ytick.minor.size: 4
xtick.major.size: 6
ytick.major.size: 6
xtick.minor.width: 0.8
ytick.minor.width: 0.8
axes.grid: True
grid.linestyle: --
grid.linewidth: 0.8
grid.alpha: 0.8
axes.facecolor : (0.95, 0.95, 0.95)
axes.prop_cycle: cycler('color', ['#da7b36', '#3ec8b2', '#0b565a', '#aebf4f', '#ef3c29', '#fbcf48', '#f7f5c4', '#21d1cb', '#016d66'])
"""

_imshow = """
font.family: 'Arial', 'SimHei'
xtick.minor.visible: True
ytick.minor.visible: True
xtick.minor.size: 4
ytick.minor.size: 4
xtick.major.size: 6
ytick.major.size: 6
xtick.minor.width: 0.8
ytick.minor.width: 0.8
axes.prop_cycle: cycler('color', ['#da7b36', '#3ec8b2', '#0b565a', '#aebf4f', '#ef3c29', '#fbcf48', '#f7f5c4', '#21d1cb', '#016d66'])
"""

_THEMES = {'spectrum': _spectrum, 'default': 'default', 'imshow': _imshow}
_FHelveticaNeueCondensedBold = {
    'family': 'Helvetica Neue',
    'weight': 700,
    'stretch': 'condensed'
}
_FHelveticaNeueCondensedBlack = {
    'family': 'Helvetica Neue',
    'weight': 900,
    'stretch': 'condensed'
}

_FONTS_PRESET = {
    'helveticaneuecondensedbold': _FHelveticaNeueCondensedBold,
    'helveticaneuecondensedblack': _FHelveticaNeueCondensedBlack,
    'hncb': _FHelveticaNeueCondensedBold,
}

import ast
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


class load_theme:

    def __init__(self, theme='default', font_zh='SimHei'):
        self.theme = _THEMES[theme]
        self.pth = None
        self.font_zh = font_zh
        self._title_font = None
        self._label_font = None
        self._tick_font = None
        self._legend_font = None
        self._minor_ticks = None
        self._using_hcnb = False

    def parse_style_string(self, style_str):
        if style_str == 'default':
            return style_str
        style_dict = {}
        for line in style_str.strip().splitlines():
            if line and not line.startswith("#"):
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                # 使用 ast.literal_eval 安全地解析 Python 表达式
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass  # 保留原字符串值
                style_dict[key] = value
        return style_dict

    def __enter__(self):
        style_file = self.parse_style_string(self.theme)

        if self._using_hcnb:
            font = FontProperties(family=['Helvetica Neue', 'Microsoft YaHei'], weight=700, stretch='condensed')
            # self._tick_font = font
            self._label_font = font
            self._title_font = font
            self._legend_font = font
            # style_file['font.family'] = ['Helvetica Neue', 'Microsoft YaHei']
            # style_file['font.weight'] = 700
            # style_file['font.stretch'] = 'condensed'

        self.pth = plt.style.context(style_file)
        self.pth.__enter__()
        if self._minor_ticks:
            set_minor_ticks(*self._minor_ticks)

    def __exit__(self, *args, **kwargs):
        self.pth.__exit__(*args, **kwargs)
        if self._title_font:
            set_title_font(self._title_font)
        if self._label_font:
            set_label_font(self._label_font)
        if self._tick_font:
            set_tick_font(self._tick_font)
        if self._legend_font:
            set_legend_font(self._legend_font)

        # set_mixed_fonts(self.font_zh)


    def set_title_font(self, fontdict):
        self._title_font = fontdict
        return self

    def set_label_font(self, fontdict):
        self._label_font = fontdict
        return self

    def set_tick_font(self, fontdict):
        self._tick_font = fontdict
        return self

    def set_legend_font(self, fontdict):
        self._legend_font = fontdict
        return self

    def set_minor_ticks(self, x=True, y=True):
        self._minor_ticks = (x, y)
        return self

    def using_hncb(self):
        self._using_hcnb = True
        return self


def set_mixed_fonts(font_zh='SimHei'):
    """
    Parameters:
    - fig: matplotlib.figure.Figure 对象
    - font_zh: FontProperties 对象，用于中文文本
    """
    fig = plt.gcf()
    for text in fig.findobj(match=plt.Text):
        is_text_in_title = any(text is ax.title for ax in fig.axes)
        if any('\u4e00' <= char <= '\u9fff' for char in text.get_text()):
            if is_text_in_title and isinstance(font_zh, str):
                current_size = text.get_fontsize()
                font_prop = FontProperties(family=font_zh,
                                           size=current_size + 2)
                text.set_fontproperties(font_prop)
            else:
                text.set_fontproperties(font_zh)


def set_title_font(fontdict):
    fontdict = _preprocess_font(fontdict)
    fig = plt.gcf()
    axes_list = fig.get_axes()
    for ax in axes_list:
        if isinstance(fontdict, dict) and  'size' not in fontdict:
            size = ax.title.get_fontsize()
            fontdict['size'] = size
        ax.title.set_fontproperties(fontdict)


def set_label_font(fontdict):
    fontdict = _preprocess_font(fontdict)
    fig = plt.gcf()
    axes_list = fig.get_axes()
    for ax in axes_list:
        if isinstance(fontdict, dict) and  'size' not in fontdict:
            size = ax.xaxis.label.get_fontsize()
            fontdict['size'] = size
        ax.xaxis.label.set_fontproperties(fontdict)
        ax.yaxis.label.set_fontproperties(fontdict)


def set_tick_font(fontdict):
    fontdict = _preprocess_font(fontdict)
    fig = plt.gcf()
    axes_list = fig.get_axes()
    for ax in axes_list:
        for tick in ax.get_xticklabels():
            if isinstance(fontdict, dict) and  'size' not in fontdict:
                size = tick.get_fontsize()
                fontdict['size'] = size
            tick.set_fontproperties(fontdict)
        for tick in ax.get_yticklabels():
            if isinstance(fontdict, dict) and  'size' not in fontdict:
                size = tick.get_fontsize()
                fontdict['size'] = size
            tick.set_fontproperties(fontdict)


def set_legend_font(fontdict):
    fontdict = _preprocess_font(fontdict)
    fig = plt.gcf()
    axes_list = fig.get_axes()
    for ax in axes_list:
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                if isinstance(fontdict, dict) and  'size' not in fontdict:
                    size = text.get_fontsize()
                    fontdict['size'] = size
                text.set_fontproperties(fontdict)


def _preprocess_font(fontdict):
    if isinstance(fontdict, str) and fontdict.lower() in _FONTS_PRESET:
        return _FONTS_PRESET[fontdict.lower()].copy()
    return fontdict


def set_minor_ticks(x=True, y=True):
    plt.rcParams['xtick.minor.visible'] = x
    plt.rcParams['ytick.minor.visible'] = y

    if x:
        plt.rcParams['xtick.minor.size'] = 4
        plt.rcParams['xtick.minor.width'] = 0.8
        plt.rcParams['xtick.major.size'] = 6

    if y:
        plt.rcParams['ytick.minor.size'] = 4
        plt.rcParams['ytick.minor.width'] = 0.8
        plt.rcParams['ytick.major.size'] = 6
