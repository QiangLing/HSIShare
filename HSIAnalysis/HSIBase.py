"""
高光谱图像处理基本函数

对高光谱图像进行处理
"""

import numpy as np

__all__ = ['HSINormalization', 'HSIConvert2D']


def HSINormalization(HSI, ImLim=(0, 1), TarDictIn=None):
    """
    高光谱图像规范化到[0, 1]

    Parameters
    ----------
    HSI
    ImLim: HSI最大值、最小值的比例范围，用做图像显示
    TarDictIn: 同时规范化的目标字典
    Returns
    -------
    规范化的HSI
    """
    minVal = np.min(HSI)
    maxVal = np.max(HSI)
    if maxVal is minVal:
        NormalizedHSI = np.zeros_like(HSI)
    else:
        minLim = minVal + (maxVal - minVal) * ImLim[0]
        maxLim = minVal + (maxVal - minVal) * ImLim[1]
        NormalizedHSI = (HSI - minLim) / (maxLim - minLim)
        NormalizedHSI[NormalizedHSI > 1] = 1
        NormalizedHSI[NormalizedHSI < 0] = 0
        if TarDictIn is not None:
            TarDict = (TarDictIn - minLim) / (maxLim - minLim)
            TarDict[TarDict > 1] = 1
            TarDict[TarDict < 0] = 0
    if TarDictIn is None:
        return NormalizedHSI
    else:
        return NormalizedHSI, TarDict


def HSIConvert2D(HSI):
    """
    高光谱图像转化为2D图像

    Parameters
    ----------
    HSI

    Returns
    -------
    每一行都是一条光谱
    """
    m, n, p = HSI.shape
    Result = np.reshape(HSI, (m * n, p))
    return Result


