"""
高光谱异常检测

Number和Lambda进行二选一赋值
说明：协方差矩阵求逆采用特征值分解方法或正则化方法。
很奇怪：OMP中的inv用numpy的运行快，用scipy中的运行慢；但CR中的inv用scipy的运行快，numpy运行慢
原因：numpy的linalg包是轻量级的，针对小矩阵，scipy的linalg包是重量级的，针对大矩阵：
（不同函数维数不一样，inv为15，svd为30，eigh为10，eig为50）
矩阵维数大于30*30的用scipy的linalg包
矩阵维数小于30*30的用numpy的linalg包
求逆然后相乘最好用solve，如：dot(inv(a), b)=solve(a, b)
"""

import numpy as np
from numpy.linalg import multi_dot
import scipy as sp
from numpy import dot, diag, eye, mean, zeros, ones, sum, sqrt
# from numpy.linalg import inv, solve # 论文中用的这个，SR、CR、KSR、KCR改用solve函数了
from scipy.linalg import inv, pinv, solve, norm  # 针对大矩阵运行速度快
from .HSIBase import *
import itertools
from sklearn.covariance import empirical_covariance, fast_mcd
from ..DataAnalysis import  ROC_AUC
from time import time
from sklearn.metrics.pairwise import pairwise_kernels
# from sklearn import svm
# from cvxopt import solvers, matrix
from ..LQ_SVM import *
from sklearn.ensemble import IsolationForest
__all__ = ['HSI_CSR_AD2']
def HSI_CSR_AD2(HSI, Windows=(5, 11), Kernel='rbf', **kwds):
    """
    基于带约束稀疏表示的异常检测

    Parameters
    ----------
    HSI
    Windows
    Kernel
    kwds

    Returns
    -------
    """
    m, n, p = HSI.shape
    semi_InW = int((Windows[0] - 1) / 2)
    semi_OutW = int((Windows[1] - 1) / 2)
    # 构造回形窗
    DW_Mask = ones((Windows[1], Windows[1]), dtype=bool)
    DW_Mask[semi_OutW - semi_InW:semi_OutW + semi_InW + 1, semi_OutW - semi_InW:semi_OutW + semi_InW + 1] = False
    # 构造HSI掩膜
    HSI_Mask = zeros((m + 2 * semi_OutW, n + 2 * semi_OutW), dtype=bool)
    HSI_Mask[semi_OutW:semi_OutW + m, semi_OutW:semi_OutW + n] = True
    # 扩展HSI
    Temp_HSI = zeros((m + 2 * semi_OutW, n + 2 * semi_OutW, p))
    Temp_HSI[semi_OutW:semi_OutW + m, semi_OutW:semi_OutW + n, :] = HSI

    # region 检测算法
    Result = zeros((m, n))
    for i, j in itertools.product(range(semi_OutW, semi_OutW + m), range(semi_OutW, semi_OutW + n)):
        Local_HSI = Temp_HSI[i - semi_OutW:i + semi_OutW + 1, j - semi_OutW:j + semi_OutW + 1, :]
        Local_Mask = HSI_Mask[i - semi_OutW:i + semi_OutW + 1, j - semi_OutW:j + semi_OutW + 1]
        Mask = Local_Mask & DW_Mask
        Background = Local_HSI[Mask, :]
        y = Temp_HSI[i, j, :]  # .reshape(1, -1)

        clf = CSR_AD2(kernel=Kernel, **kwds)
        dec = clf.decision_function(Background, y)


        Result[i - semi_OutW, j - semi_OutW] = dec
    # endregion

    return Result


