# Copyright (c) 2024 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Some default colors
"""

c2 = [
    ['#62B197', '#E18E6D'],
    ['#B8DDBC', '#F0A780'],
    ['#97C8AF', '#96B6D8'],
    ['#8FC9E2', '#ECC97F'],
    ['#D6AFB9', '#7E9BB7'],
    ['#F89FA8', '#F9E9A4'],
    ['#EE8883', '#8DB7DB'],
]

c3 = [
    ['#9392BE', '#D0E7ED', '#D5E4A8'],
    ['#F1C89A', '#E79397', '#A797DA'],
    ['#E1C855', '#E07B54', '#51B1B7'],  # line
    ['#A5C496', '#C7988C', '#8891DB'],
    ['#0E986F', '#796CAD', '#D65813'],
    ['#A9CA70', '#C5D6F0', '#F18C54'],
    ['#377EB9', '#4DAE48', '#974F9F'],
]

c4 = [
    ['#9BC985', '#F7D58B', '#B595BF', '#797BB7'],
    ['#F50804', '#9925E1', '#BDBDBD', '#000000'],
    ['#42B796', '#4394C4', '#EDBA42', '#D7D7D7'],
    # ['#', '#', '#', '#'],
]

c5 = [
    ['#C6B3D3', '#ED9F9B', '#80BA8A', '#9CD1C8', '#6BB7CA'],
    ['#9DD0C7', '#9180AC', '#D9BDD8', '#E58579', '#8AB1D2'],
    ['#EEC79F', '#F1DFA4', '#74B69F', '#A6CDE4', '#E2C8D8'],
    ['#CC88B0', '#998DB7', '#DBE0ED', '#87B5B2', '#F4CEB4'],
    ['#F1DBE7', '#E0F1F7', '#DBD8E9', '#DEECD9', '#D0D2D4'],  # 淡色, 用于边缘填充
    ['#2878b5', '#9ac9db', '#f8ac8c', '#c82423', '#ff8884'],  # Marked! line
]

c6 = [
    ['#979AAA', '#D4AE9D', '#6FA9B5', '#3F4B69', '#985B54', '#208974'],
]

# fmt: off
c7 = [
    ['#818181', '#295522', '#66B543', '#E07E35', '#F2CCA0', '#A9C4E6', '#D1392B'],
    ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#BEB8DC', '#E7DAD2', '#999999'],
    ['#F27970', '#BB9727', '#54B345', '#32B897', '#05B9E2', '#8983BF', '#C76DA2'],

]


c8 = [
    ['#A1A9D0', '#F0988C', '#B883D4', '#9E9E9E', '#CFEAF1', '#C4A5DE', '#F6CAE5', '#96CCCB'],
]




import matplotlib.pyplot as plt

def view_colors(colors):
    """
    Visualize a given list of colors and test it with the following graphic:
    1. Color Block 
    2. Line
    3. Bar
    4. Scatter
    5. Pie chart
    6. Histogram
    
    Parameters:
    colors (list): List of color codes, e.g., ['#9392BE', '#D0E7ED', '#D5E4A8']。
    """
    n = len(colors)

    fig, axs = plt.subplots(3, 2, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.5)

    axs[0, 0].axis('off')
    for i, color in enumerate(colors):
        axs[0, 0].add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
        axs[0, 0].text(i/n+0.5/n, 0.3, color, ha='center', va='top', fontsize=10, transform=axs[0, 0].transAxes, fontweight='bold')
    axs[0, 0].set_xlim(0, n)
    axs[0, 0].set_ylim(-0.5, 1)
    axs[0, 0].set_title('Color Blocks')

    for i, color in enumerate(colors):
        axs[0, 1].plot(range(10), [x + i for x in range(10)], label=f'Line {i+1}', color=color, linewidth=2)
    axs[0, 1].legend(loc='best')
    axs[0, 1].set_title('Line Plot')

    bars = axs[1, 0].bar(range(n), [i + 1 for i in range(n)], color=colors, tick_label=[f'Bar {i+1}' for i in range(n)])
    axs[1, 0].bar_label(bars)
    axs[1, 0].set_title('Bar Plot')
    axs[1, 0].set_ylim(0, n+0.5)

    for i, color in enumerate(colors):
        axs[1, 1].scatter(range(10), [x + i for x in range(10)], color=color, s=50, label=f'Scatter {i+1}')
    axs[1, 1].legend(loc='best')
    axs[1, 1].set_title('Scatter Plot')

    axs[2, 0].pie([1] * n, labels=[f'Pie {i+1}' for i in range(n)], colors=colors, autopct='%1.1f%%')
    axs[2, 0].set_title('Pie Chart')

    data = [[i + j for j in range(10)] for i in range(n)]
    axs[2, 1].hist(data, bins=10, color=colors, label=[f'Hist {i+1}' for i in range(n)])
    axs[2, 1].legend(loc='best')
    axs[2, 1].set_title('Histogram')

    plt.tight_layout()
    plt.show()
