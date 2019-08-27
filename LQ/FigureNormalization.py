"""
标准绘图模块

import模块会导入模块里定义的函数、变量以及模块里导入的模块
提供规范化绘图参数、规范化绘图函数等（如PlotNormalization）
'SimSun' 宋体
'SimHei' 黑体
'KaiTi' 楷体
'FangSong' 仿宋
'STSong' 华文宋体
"""

import matplotlib.pyplot as plt
from .HSIAnalysis import HSINormalization
import numpy as np
import scipy as sp
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from cycler import cycler
from enum import Enum
import itertools
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import axes3d
import os

__all__ = ['plt', 'np', 'sp', 'Size', 'rcParams', 'PlotNormalization', 'PlotROC', 'ImshowNormalization',
           'ImSurfNormalization', 'SurfaceNormalization', 'PlotConfusionMatrix', 'ImCut', 'MyFont']

# region 绘图参数rcParams设置
MyFont = FontProperties(family='SimHei, Times New Roman')  # 黑体 SimHei SimSun
rcParams['backend'] = 'Qt5Agg'
# rcParams['backend.qt5'] = 'PyQt5'
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams['font.size'] = 16.0
rcParams['mathtext.rm'] = 'Times New Roman'
rcParams['mathtext.it'] = 'Times New Roman:italic'
rcParams['mathtext.bf'] = 'Times New Roman:bold'
rcParams['axes.titlesize'] = 'medium'
rcParams['axes.unicode_minus'] = False
rcParams['axes.grid'] = True
rcParams['axes.linewidth'] = 0.6
rcParams['savefig.dpi'] = 600
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.01  # 0.05
rcParams['grid.alpha'] = 0.7
rcParams['legend.fontsize'] = 'medium'
rcParams['legend.loc'] = 'best'
rcParams['legend.frameon'] = True
rcParams['lines.markersize'] = 3.0
rcParams['lines.markeredgewidth'] = 0.1
rcParams['xtick.minor.visible'] = False
rcParams['xtick.major.size'] = 3.0
rcParams['xtick.minor.size'] = 1.5
rcParams['ytick.minor.visible'] = False
rcParams['ytick.major.size'] = 3.0
rcParams['ytick.minor.size'] = 1.5
rcParams['axes.prop_cycle'] = cycler('color', ['b', 'r', 'g', 'm', 'c', 'y', 'k']) \
                              + cycler('marker', ['o', 's', 'p', '^', '*', 'h', 'H'])


# endregion


class Size(Enum):
    VerySmall = 0
    Small = 1
    Medium = 2
    Large = 3
    VeryLarge = 4


def PlotNormalization(XLabel, YLabel, LegendLabel=None, Title=None, FigName=None, FigSize=Size.Medium, WindowTitle=None,
                      LegendLoc='best'):
    """
    Plot函数作图规范化
    Parameters
    ----------
    XLabel
    YLabel
    LegendLabel
    Title
    FigName: 保存图片的名称
    FigSize: 保存图片的大小，枚举Size类型。可设为：'Large'、'Small'、'Medium'
    WindowTitle: 窗口标题
    LegendLoc:
    Returns
    -------

    """

    rcParams['font.size'] = 7.5
    ax = plt.gca()
    xticks = ax.get_xticks()
    nonzeroxticks = xticks[np.nonzero(xticks)]
    yticks = ax.get_yticks()
    nonzeroyticks = yticks[np.nonzero(yticks)]
    texts = []
    if ax.get_xscale() is not 'log':
        if np.abs(xticks).max() >= 1e4:
            temp_ticks = np.abs(xticks).max() / 10
            num = 1
            while temp_ticks >= 10:
                temp_ticks /= 10
                num += 1
            xticks_new = xticks / (10 ** num)
            ax.set_xticklabels(xticks_new)
            # pos_x = np.max(xticks) + 0.005*(xticks.max() - xticks.min())
            # pos_y = np.min(yticks)
            # text1 = ax.text(pos_x, pos_y, '$\\times 10^%d$' % num, fontsize=7)
            text1 = plt.gcf().text(0.905, 0.1, '$\\times 10^%d$' % num, fontsize=7)
            texts.append(text1)

        if np.abs(nonzeroxticks).min() <= 1e-3:
            temp_ticks = np.abs(nonzeroxticks).min() * 10
            num = 1
            while temp_ticks <= 0.1:
                temp_ticks *= 10
                num += 1
            xticks_new = xticks * (10 ** num)
            ax.set_xticklabels(xticks_new)
            # pos_x = np.max(xticks) + 0.005*(xticks.max() - xticks.min())
            # pos_y = np.min(yticks)
            # text1 = ax.text(pos_x, pos_y, '$\\times 10^{-%d}$' % num, fontsize=7)
            text1 = plt.gcf().text(0.905, 0.1, '$\\times 10^{-%d}$' % num, fontsize=7)
            texts.append(text1)

    if ax.get_yscale() is not 'log':
        if np.abs(yticks).max() >= 1e4:
            temp_ticks = np.abs(yticks).max() / 10
            num = 1
            while temp_ticks >= 10:
                temp_ticks /= 10
                num += 1
            yticks_new = yticks / (10 ** num)
            ax.set_yticklabels(yticks_new)
            # pos_x = np.min(xticks)
            # pos_y = np.max(yticks) + 0.01 * (yticks.max() - yticks.min())
            # text2 = ax.text(pos_x, pos_y, '$\\times 10^%d$' % num, fontsize=7)
            text2 = plt.gcf().text(0.12, 0.91, '$\\times 10^%d$' % num, fontsize=7)
            texts.append(text2)

        if np.abs(nonzeroyticks).min() <= 1e-3:
            temp_ticks = np.abs(nonzeroyticks).min() * 10
            num = 1
            while temp_ticks <= 0.1:
                temp_ticks *= 10
                num += 1
            yticks_new = yticks * (10 ** num)
            ax.set_yticklabels(yticks_new)
            # pos_x = np.min(xticks)
            # pos_y = np.max(yticks) + 0.01 * (yticks.max() - yticks.min())
            # text2 = ax.text(pos_x, pos_y, '$\\times 10^{-%d}$' % num, fontsize=7)
            text2 = plt.gcf().text(0.12, 0.91, '$\\times 10^{-%d}$' % num, fontsize=7)
            texts.append(text2)

    X_Index = [1 for char in XLabel if '\u4e00' <= char <= '\u9fa5']
    if any(X_Index):
        plt.xlabel(XLabel, fontproperties=MyFont)  # 汉字用黑体 宋体 SimSun
    else:
        plt.xlabel(XLabel)

    Y_Index = [1 for char in YLabel if '\u4e00' <= char <= '\u9fa5']
    if any(Y_Index):
        plt.ylabel(YLabel, fontproperties=MyFont)  # 汉字用黑体
    else:
        plt.ylabel(YLabel)

    if LegendLabel is not None:
        Legend = plt.legend(LegendLabel, loc=LegendLoc)  # upper right LegendLoc  , bbox_to_anchor=(1.03, 1.035)
        for LegendString in Legend.get_texts():
            Legend_Index = [1 for char in LegendString.get_text() if '\u4e00' <= char <= '\u9fa5']
            if any(Legend_Index):
                LegendString.set(fontproperties=MyFont)
        Legend.get_frame().set_linewidth(0.5)

    if Title is not None:
        T_Index = [1 for char in Title if '\u4e00' <= char <= '\u9fa5']
        if any(T_Index):
            plt.title(Title, fontproperties=MyFont)  # 汉字用黑体 SimSun
        else:
            plt.title(Title)

    fig = plt.gcf()
    if FigSize is Size.VerySmall:
        rcParams['savefig.pad_inches'] = 0.01  # 0.01
        fig.set_size_inches(1.5, 1.3)
    elif FigSize is Size.Small:
        rcParams['savefig.pad_inches'] = 0.01  # 0.03
        fig.set_size_inches(1.8, 1.6)  # 2.0, 1.8  1.8, 1.6
    elif FigSize is Size.Medium:
        rcParams['savefig.pad_inches'] = 0.01  # 0.05
        fig.set_size_inches((3, 2.5))  # 2.8, 2.3
    elif FigSize is Size.Large:
        rcParams['savefig.pad_inches'] = 0.01  # 0.07
        fig.set_size_inches(4.0, 3.2)  # 5.0, 4.0
    elif FigSize is Size.VeryLarge:
        rcParams['savefig.pad_inches'] = 0.01  # 0.09
        fig.set_size_inches(6.0, 4.8)
    else:
        raise ValueError("FigSize must be a enum object 'Size'!")

    if FigName is not None:
        if not os.path.exists('Figs'):
            os.mkdir('Figs')
        fig.savefig('Figs\\'+FigName)

    if WindowTitle is not None:
        fig.canvas.manager.set_window_title(WindowTitle)
    elif FigName is not None:
        fig.canvas.manager.set_window_title(FigName)

    [text.set(fontsize=15) for text in texts]
    rcParams['font.size'] = 16.0
    rcParams['savefig.pad_inches'] = 0.01  # 0.05
    return


def PlotROC(Pf, Pd, CI=False, FigName=None, xlim=(1e-4, 1.0), ylim=(0.0, 1.0), markersize=0.0, LegendLabel=None, LegendLoc='upper left', **kwargs):
    """

    Parameters
    ----------
    Pf
    Pd
    CI: 是否画Pf的置信区间
    FigName
    xlim
    ylim
    markersize: markersize of the center ROC curve
    LegendLabel
    LegendLoc

    Returns
    -------

    """
    rcParams['font.size'] = 7.5
    color = ['r', 'b', 'g', 'm', 'c', 'k', 'y', 'chartreuse', 'burlywood']
    marker = ['o', 's', 'p', '^', '*', 'h', 'H', 'v', '>']
    N = len(Pd)
    line = []
    for i in range(N):
        if CI is True:
            plt.plot(Pf[N - 1 - i][0], Pd[N - 1 - i], color=color[i], marker=marker[i], linewidth=0.3, linestyle='solid')
            tmp, = plt.plot(Pf[N - 1 - i][1], Pd[N - 1 - i], color=color[i], marker=marker[i], linewidth=0.8, markersize=markersize)
            line.append(tmp)
            plt.plot(Pf[N - 1 - i][2], Pd[N - 1 - i], color=color[i], marker=marker[i], linewidth=0.3, linestyle='solid')
        else:
            tmp, = plt.plot(Pf[N - 1 - i], Pd[N - 1 - i], color=color[i], marker=marker[i])
            line.append(tmp)

    if LegendLabel is not None:
        Legend = plt.legend(line[::-1], LegendLabel, loc=LegendLoc)
        for LegendString in Legend.get_texts():
            Legend_Index = [1 for char in LegendString.get_text() if '\u4e00' <= char <= '\u9fa5']
            if any(Legend_Index):
                LegendString.set(fontproperties=MyFont)
        Legend.get_frame().set_linewidth(0.5)

    plt.xscale('log')
    plt.xlim(xlim)
    plt.ylim(ylim)
    XLabel = 'False alarm rate'
    YLabel = 'Probability of detection'
    PlotNormalization(XLabel, YLabel, FigName=FigName, **kwargs)
    return


def ImshowNormalization(OriginalIm, ImLim=None, Cmap='gray', AxisOn=False, Title=None, ColorBarOn=False, FigName=None,
                        FigSize=Size.Medium, WindowTitle=None):
    """
    Imshow函数作图规范化(显示和规范)

    Parameters
    ----------
    OriginalIm: 原图像
    ImLim: 显示范围
    Cmap
    AxisOn: 是否画坐标轴
    Title
    ColorBarOn: 是否画颜色条
    FigName: 保存图片的名称
    FigSize: 保存图片的大小，枚举Size类型。可设为：'Large'、'Small'、'Medium'
    WindowTitle:
    Returns
    -------

    """
    rcParams['font.size'] = 7.5
    rcParams['axes.grid'] = False
    if ImLim is not None:
        OriginalIm = HSINormalization(OriginalIm, ImLim=ImLim)
    if OriginalIm.ndim is 3:
        OriginalIm = HSINormalization(OriginalIm)
    plt.imshow(OriginalIm, cmap=Cmap, interpolation='nearest')
    if Title is not None:
        T_Index = [1 for char in Title if '\u4e00' <= char <= '\u9fa5']
        if any(T_Index):
            plt.title(Title, fontproperties=MyFont)  # 汉字用黑体
        else:
            plt.title(Title)

    if ColorBarOn:
        plt.colorbar(shrink=0.75)

    fig = plt.gcf()
    if FigSize is Size.VerySmall:
        rcParams['savefig.pad_inches'] = 0.01  # 0.01
        fig.set_size_inches(1.5, 1.5)
    elif FigSize is Size.Small:
        rcParams['savefig.pad_inches'] = 0.01  # 0.03
        fig.set_size_inches(2.0, 2.0)
    elif FigSize is Size.Medium:
        rcParams['savefig.pad_inches'] = 0.01  # 0.05
        fig.set_size_inches(3.0, 3.0)
    elif FigSize is Size.Large:
        rcParams['savefig.pad_inches'] = 0.01  # 0.07
        fig.set_size_inches(5.0, 5.0)
    elif FigSize is Size.VeryLarge:
        rcParams['savefig.pad_inches'] = 0.01  # 0.09
        fig.set_size_inches(20.0, 10.0)
    else:
        raise ValueError("FigSize must be a enum object 'Size'!")

    if not AxisOn:
        plt.gca().set_axis_off()

    if FigName is not None:
        if not AxisOn and not ColorBarOn:
            # pad_inches表示字和白边的距离，这里可用dpi控制图像大小，size=len(OriginalIm)/DPI*2.54（cm）
            rcParams['savefig.pad_inches'] = 0
            if FigSize is Size.VerySmall:
                DPI = len(OriginalIm) / 2.8 * 2.54  # 2.8cm
            elif FigSize is Size.Small:
                DPI = len(OriginalIm) / 4.8 * 2.54  # 4.8cm
            elif FigSize is Size.Medium:
                DPI = len(OriginalIm) / 7 * 2.54  # 7cm
            elif FigSize is Size.Large:
                DPI = len(OriginalIm) / 9 * 2.54  # 9cm
            elif FigSize is Size.VeryLarge:
                DPI = len(OriginalIm) / 13 * 2.54  # 13cm
            else:
                raise ValueError("FigSize must be a enum object 'Size'!")

            if not os.path.exists('Figs'):
                os.mkdir('Figs')
            if rcParams['savefig.format'] == 'eps' or FigName[-3:] == 'eps':
                plt.imsave('Figs\\' + FigName, OriginalIm, cmap=Cmap)
            else:
                plt.imsave('Figs\\' + FigName, OriginalIm, cmap=Cmap, dpi=DPI)
        else:
            if not os.path.exists('Figs'):
                os.mkdir('Figs')
            fig.savefig('Figs\\' + FigName)

    if WindowTitle is not None:
        fig.canvas.manager.set_window_title(WindowTitle)
    elif FigName is not None:
        fig.canvas.manager.set_window_title(FigName)

    rcParams['font.size'] = 16.0
    rcParams['axes.grid'] = True
    rcParams['savefig.pad_inches'] = 0.01  # 0.05
    return


def ImSurfNormalization(OriginalIm, Cmap='jet', Title=None, ColorBarOn=False, FigName=None, FigSize=Size.Medium,
                        Height_Width=0.8, WindowTitle=None):
    """
    surface绘制图片规范化
    Parameters
    ----------
    OriginalIm: 原图像
    Cmap
    Title
    ColorBarOn: 颜色条
    FigName: 保存的图片名称
    FigSize: 保存的图片大小
    Height_Width: 保存图片高度和宽度的比值
    WindowTitle
    Returns
    -------

    """
    rcParams['savefig.bbox'] = None
    X = np.arange(0, OriginalIm.shape[0])
    Y = np.arange(0, OriginalIm.shape[1])
    X, Y = np.meshgrid(X, Y)
    fig = plt.gcf()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X.T, Y.T, OriginalIm, cmap=Cmap, linewidth=0.05)
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    zticks = ax.get_zticks()
    if np.abs(xticks).max() >= 1e3:
        temp_ticks = np.abs(xticks).max() / 10
        num = 1
        while temp_ticks >= 10:
            temp_ticks /= 10
            num += 1
        xticks_new = np.array(xticks / (10 ** num), dtype=str)
        xticks_new[-1] += '$\\times 10^%d$' % num
        ax.set_xticklabels(xticks_new)

    if np.abs(yticks).max() >= 1e3:
        temp_ticks = np.abs(yticks).max() / 10
        num = 1
        while temp_ticks >= 10:
            temp_ticks /= 10
            num += 1
        yticks_new = np.array(yticks / (10 ** num), dtype=str)
        yticks_new[-1] += '$\\times 10^%d$' % num
        ax.set_yticklabels(yticks_new)

    if np.abs(zticks).max() >= 1e3:
        temp_ticks = np.abs(zticks).max() / 10
        num = 1
        while temp_ticks >= 10:
            temp_ticks /= 10
            num += 1
        zticks_new = np.array(zticks / (10 ** num), dtype=str)
        zticks_new[-1] += '$\\times 10^%d$' % num
        ax.set_zticklabels(zticks_new)

    if Title is not None:
        T_Index = [1 for char in Title if '\u4e00' <= char <= '\u9fa5']
        if any(T_Index):
            plt.title(Title, fontproperties=MyFont)  # 汉字用黑体
        else:
            plt.title(Title)

    if ColorBarOn:
        Height_Width = 0.7
        fig.colorbar(surf, shrink=0.8)

    fig.set_size_inches(5.0, 5.0 * Height_Width)
    if FigSize is Size.VerySmall:
        rcParams['font.size'] = 12
    elif FigSize is Size.Small:
        rcParams['font.size'] = 12
    elif FigSize is Size.Medium:
        rcParams['font.size'] = 9
    elif FigSize is Size.Large:
        rcParams['font.size'] = 7.5
    elif FigSize is Size.VeryLarge:
        rcParams['font.size'] = 6
    else:
        raise ValueError("FigSize must be a enum object 'Size'!")

    if FigName is not None:
        # from matplotlib.transforms import Bbox
        # OriginalBox = np.array(fig.bbox_inches)
        # TightBox = Bbox(OriginalBox + [[0.55, 0.05], [-0.15, -0.35]])
        # fig.savefig(FigName, bbox_inches='tight', pad_inches=0.1)
        if not os.path.exists('Figs'):
            os.mkdir('Figs')
        fig.savefig('Figs\\' + FigName)
        Im = plt.imread('Figs\\' + FigName + '.png')
        CutIm = ImCut(Im)
        rcParams['savefig.pad_inches'] = 0
        if FigSize is Size.VerySmall:
            DPI = len(CutIm) / 2.6 * 2.54  # 2.8cm
        elif FigSize is Size.Small:
            DPI = len(CutIm) / 4.2 * 2.54  # 4.8cm
        elif FigSize is Size.Medium:
            DPI = len(CutIm) / 6.5 * 2.54  # 7cm
        elif FigSize is Size.Large:
            DPI = len(CutIm) / 8.2 * 2.54  # 9cm
        elif FigSize is Size.VeryLarge:
            DPI = len(CutIm) / 12 * 2.54  # 13cm
        else:
            raise ValueError("FigSize must be a enum object 'Size'!")

        if rcParams['savefig.format'] == 'eps' or FigName[-3:] == 'eps':
            plt.imsave('Figs\\' + FigName, CutIm, cmap=Cmap)
        else:
            plt.imsave('Figs\\' + FigName, CutIm, cmap=Cmap, dpi=DPI)

    if WindowTitle is not None:
        fig.canvas.manager.set_window_title(WindowTitle)
    elif FigName is not None:
        fig.canvas.manager.set_window_title(FigName)

    rcParams['font.size'] = 16.0
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.pad_inches'] = 0.01  # 0.05
    return


def SurfaceNormalization(X, Y, Z, XLabel=None, YLabel=None, ZLabel=None, rstride=10, cstride=10, LabelPad=0.08,
                         Cmap='jet', Title=None, ColorBarOn=False, FigName=None, FigSize=Size.Medium, Height_Width=0.8,
                         WindowTitle=None):
    """
    surface绘制3D图规范化
    Parameters
    ----------
    X
    Y
    Z
    XLabel
    YLabel
    ZLabel
    rstride: x轴绘制间隔
    cstride: y轴绘制间隔
    LabelPad
    Cmap
    Title
    ColorBarOn
    FigName
    FigSize
    Height_Width: 保存图像高度和宽度的比值
    WindowTitle
    Returns
    -------

    """
    rcParams['savefig.bbox'] = None
    fig = plt.gcf()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=rstride, cstride=cstride, cmap=Cmap, linewidth=0.1)

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    zticks = ax.get_zticks()
    if np.abs(xticks).max() >= 1e3:
        temp_ticks = np.abs(xticks).max() / 10
        num = 1
        while temp_ticks >= 10:
            temp_ticks /= 10
            num += 1
        xticks_new = np.array(xticks / (10 ** num), dtype=str)
        xticks_new[-1] += '$\\times 10^%d$' % num
        ax.set_xticklabels(xticks_new)

    if np.abs(yticks).max() >= 1e3:
        temp_ticks = np.abs(yticks).max() / 10
        num = 1
        while temp_ticks >= 10:
            temp_ticks /= 10
            num += 1
        yticks_new = np.array(yticks / (10 ** num), dtype=str)
        yticks_new[-1] += '$\\times 10^%d$' % num
        ax.set_yticklabels(yticks_new)

    if np.abs(zticks).max() >= 1e3:
        temp_ticks = np.abs(zticks).max() / 10
        num = 1
        while temp_ticks >= 10:
            temp_ticks /= 10
            num += 1
        zticks_new = np.array(zticks / (10 ** num), dtype=str)
        zticks_new[-1] += '$\\times 10^%d$' % num
        ax.set_zticklabels(zticks_new)

    if XLabel is not None:
        X_Index = [1 for char in XLabel if '\u4e00' <= char <= '\u9fa5']
        if any(X_Index):
            ax.set_xlabel(XLabel, labelpad=LabelPad, fontproperties=MyFont)  # 汉字用黑体
        else:
            ax.set_xlabel(XLabel, labelpad=LabelPad)

    if YLabel is not None:
        Y_Index = [1 for char in YLabel if '\u4e00' <= char <= '\u9fa5']
        if any(Y_Index):
            ax.set_ylabel(YLabel, labelpad=LabelPad, fontproperties=MyFont)  # 汉字用黑体
        else:
            ax.set_ylabel(YLabel, labelpad=LabelPad)

    if ZLabel is not None:
        Z_Index = [1 for char in ZLabel if '\u4e00' <= char <= '\u9fa5']
        if any(Z_Index):
            ax.set_zlabel(ZLabel, labelpad=LabelPad, fontproperties=MyFont)  # 汉字用黑体
        else:
            ax.set_zlabel(ZLabel, labelpad=LabelPad)

    if Title is not None:
        T_Index = [1 for char in Title if '\u4e00' <= char <= '\u9fa5']
        if any(T_Index):
            plt.title(Title, fontproperties=MyFont)  # 汉字用黑体
        else:
            plt.title(Title)

    if ColorBarOn:
        Height_Width = 0.7
        fig.colorbar(surf, shrink=0.8)

    fig.set_size_inches(5.0, 5.0 * Height_Width)
    if FigSize is Size.VerySmall:
        rcParams['font.size'] = 12
    elif FigSize is Size.Small:
        rcParams['font.size'] = 12
    elif FigSize is Size.Medium:
        rcParams['font.size'] = 8.5
    elif FigSize is Size.Large:
        rcParams['font.size'] = 7.5
    elif FigSize is Size.VeryLarge:
        rcParams['font.size'] = 6
    else:
        raise ValueError("FigSize must be a enum object 'Size'!")

    if FigName is not None:
        # from matplotlib.transforms import Bbox
        # OriginalBox = np.array(fig.bbox_inches)
        # TightBox = Bbox(OriginalBox + [[0.55, 0.05], [-0.15, -0.35]])
        # fig.savefig(FigName, bbox_inches='tight', pad_inches=0.1)
        if not os.path.exists('Figs'):
            os.mkdir('Figs')
        fig.savefig('Figs\\' + FigName)
        Im = plt.imread('Figs\\' + FigName + '.png')
        CutIm = ImCut(Im)
        rcParams['savefig.pad_inches'] = 0
        if FigSize is Size.VerySmall:
            DPI = len(CutIm) / 2.6 * 2.54  # 2.8cm
        elif FigSize is Size.Small:
            DPI = len(CutIm) / 4.2 * 2.54  # 4.8cm
        elif FigSize is Size.Medium:
            DPI = len(CutIm) / 6.3 * 2.54  # 7cm
        elif FigSize is Size.Large:
            DPI = len(CutIm) / 8.2 * 2.54  # 9cm
        elif FigSize is Size.VeryLarge:
            DPI = len(CutIm) / 12 * 2.54  # 13cm
        else:
            raise ValueError("FigSize must be a enum object 'Size'!")

        # plt.imsave(FigName, OriginalIm, cmap=Cmap, dpi=DPI)
        plt.imsave('Figs\\' + FigName, CutIm, cmap=Cmap, dpi=DPI)

    if WindowTitle is not None:
        fig.canvas.manager.set_window_title(WindowTitle)
    elif FigName is not None:
        fig.canvas.manager.set_window_title(FigName)

    rcParams['font.size'] = 16.0
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.pad_inches'] = 0.01  # 0.05
    return


def PlotConfusionMatrix(ConfusionMatrix, ClassNames, Title=None, Language='EN', Cmap='Blues', FigName=None,
                        FigSize=Size.Medium, WindowTitle=None, Normalized=False):
    """
    画混淆矩阵图
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    rcParams['font.size'] = 7.5
    rcParams['axes.grid'] = False
    rcParams['ytick.major.width'] = 0.0
    rcParams['xtick.major.width'] = 0.0
    plt.imshow(ConfusionMatrix, interpolation='nearest', cmap=Cmap)
    if Title is not None:
        T_Index = [1 for char in Title if '\u4e00' <= char <= '\u9fa5']
        if any(T_Index):
            plt.title(Title, fontproperties=MyFont)  # 汉字用黑体
        else:
            plt.title(Title)

    # plt.colorbar()
    tick_marks = np.arange(len(ClassNames))
    plt.xticks(tick_marks, ClassNames, rotation=0)
    plt.yticks(tick_marks, ClassNames, rotation=90)

    plt.hlines(tick_marks[:-1] + 0.5, tick_marks[0] - 0.5, tick_marks[-1] + 0.5, linewidth=0.5, alpha=0.5)
    plt.vlines(tick_marks[:-1] + 0.5, tick_marks[0] - 0.5, tick_marks[-1] + 0.5, linewidth=0.5, alpha=0.5)

    # region 设置XLabel、YLabel
    if Language is 'EN':
        XLabel = 'Predicted label'
        YLabel = 'True label'
    elif Language is 'CN':
        XLabel = '预测类别'
        YLabel = '真实类别'
    else:
        raise ValueError("Language must be 'EN' or 'CN'!")

    X_Index = [1 for char in XLabel if '\u4e00' <= char <= '\u9fa5']
    if any(X_Index):
        plt.xlabel(XLabel, fontproperties=MyFont)  # 汉字用黑体
    else:
        plt.xlabel(XLabel)

    Y_Index = [1 for char in YLabel if '\u4e00' <= char <= '\u9fa5']
    if any(Y_Index):
        plt.ylabel(YLabel, fontproperties=MyFont)  # 汉字用黑体
    else:
        plt.ylabel(YLabel)
    # endregion

    print('Confusion Matrix')
    print(ConfusionMatrix)
    if Normalized:
        NormalizedConfusionMatrix = np.multiply(100, ConfusionMatrix) / ConfusionMatrix.sum(axis=1)[:,
                                                                        np.newaxis]  # 某一类的百分比
        print('Normalized Confusion Matrix(percent)')
        print(np.round(NormalizedConfusionMatrix, 2))

        thresh = ConfusionMatrix.max() / 2.
        text_list = []
        for i, j in itertools.product(range(ConfusionMatrix.shape[0]), range(ConfusionMatrix.shape[1])):
            text1 = plt.text(j, i - 0.12, ConfusionMatrix[i, j], horizontalalignment="center",
                             verticalalignment="center", color="white" if
                ConfusionMatrix[i, j] > thresh else "black")
            percent_text = '%0.2f' % NormalizedConfusionMatrix[i, j] + '%'
            text2 = plt.text(j, i + 0.12, percent_text, horizontalalignment="center",
                             verticalalignment="center", color="white" if
                ConfusionMatrix[i, j] > thresh else "black")
            text_list.append(text1)
            text_list.append(text2)
    else:
        thresh = ConfusionMatrix.max() / 2.
        text_list = []
        for i, j in itertools.product(range(ConfusionMatrix.shape[0]), range(ConfusionMatrix.shape[1])):
            text = plt.text(j, i, ConfusionMatrix[i, j], horizontalalignment="center",
                            verticalalignment="center", color="white" if
                ConfusionMatrix[i, j] > thresh else "black")
            text_list.append(text)

    fig = plt.gcf()
    if FigSize is Size.VerySmall:
        rcParams['savefig.pad_inches'] = 0.01  # 0.01
        fig.set_size_inches(1.5, 1.5)
    elif FigSize is Size.Small:
        rcParams['savefig.pad_inches'] = 0.01  # 0.03
        fig.set_size_inches(2.0, 2.0)
    elif FigSize is Size.Medium:
        rcParams['savefig.pad_inches'] = 0.01  # 0.05
        fig.set_size_inches(3.0, 3.0)
    elif FigSize is Size.Large:
        rcParams['savefig.pad_inches'] = 0.01  # 0.07
        fig.set_size_inches(5.0, 5.0)
    elif FigSize is Size.VeryLarge:
        rcParams['savefig.pad_inches'] = 0.01  # 0.09
        fig.set_size_inches(6.0, 6.0)
    else:
        raise ValueError("FigSize must be a enum object 'Size'!")
    if FigName is not None:
        if not os.path.exists('Figs'):
            os.mkdir('Figs')
        fig.savefig('Figs\\' + FigName)

    if WindowTitle is not None:
        fig.canvas.manager.set_window_title(WindowTitle)
    elif FigName is not None:
        fig.canvas.manager.set_window_title(FigName)

    [text.set(fontsize=16.0) for text in text_list]
    rcParams['ytick.major.width'] = 0.5
    rcParams['xtick.major.width'] = 0.5
    rcParams['font.size'] = 16.0
    rcParams['axes.grid'] = True
    return


def ImCut(OriginalIm):
    """
    裁剪图像白边
    Parameters
    ----------
    OriginalIm: 原图像

    Returns
    -------
    裁剪图像
    """
    from skimage.color import rgb2gray
    GrayIm = rgb2gray(OriginalIm)
    RIndex = np.zeros(2, dtype=int)
    CIndex = np.zeros(2, dtype=int)
    for Row in GrayIm:
        if np.mean(Row) == 1:
            RIndex[0] += 1
        else:
            break

    for Row in GrayIm[::-1]:
        if np.mean(Row) == 1:
            RIndex[1] += 1
        else:
            break

    for Column in GrayIm.T:
        if np.mean(Column) == 1:
            CIndex[0] += 1
        else:
            break

    for Column in GrayIm.T[::-1]:
        if np.mean(Column) == 1:
            CIndex[1] += 1
        else:
            break
    pad = 2
    Result = OriginalIm[RIndex[0] - pad:-RIndex[1] + pad, CIndex[0] - pad:-CIndex[1] + pad, :]
    return Result


def __main():
    """
    测试函数，包括LQ包里的PlotNormalization、ImshowNormalization、ImSurfNormalization、SurfaceNormalization等函数的用法
    """
    # MyFont = {'family': MyFont, 'size': 7.5}
    from skimage import data
    x = np.arange(0, 10)
    y = np.random.randn(10, 7)
    A = data.camera()
    cnf_mat = np.random.randint(0, 100, (5, 5))
    plt.figure(1)
    PlotConfusionMatrix(cnf_mat, np.arange(5), FigName='cnf_mat', Normalized=True)
    plt.figure(2)
    plt.plot(x, y)
    PlotNormalization('x', 'y', ['1', '2', '3', '4', '5', '6', '7'], FigName='lq', FigSize=Size.Medium)
    plt.figure(3)
    ImshowNormalization(A, FigName='Camera', FigSize=Size.Medium, AxisOn=True)
    plt.figure(4)
    ImSurfNormalization(A, FigSize=Size.Medium, FigName='3D图像', ColorBarOn=True, Title=None)
    plt.figure(5)
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)
    SurfaceNormalization(X, Y, Z, XLabel='$\phi_\mathrm{real}$', YLabel='黑体safdfsd', ZLabel='sadfasdf', rstride=1,
                         cstride=1, FigName='3DSurf', FigSize=Size.Medium, ColorBarOn=True)
    plt.show()


if __name__ == '__main__':
    print(__doc__)
    # Examples are in main() function
    print(__main.__doc__)
    __main()
