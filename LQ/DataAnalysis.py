"""
标准数据分析模块

数据分析标准模块，包含常用的数据分析模块
import模块会导入模块里定义的函数、变量以及模块里导入的模块
提供常用的数据分析函数
numpy的linalg包是轻量级的，针对小矩阵，scipy的linalg包是重量级的，针对大矩阵：
（不同函数维数不一样，inv为15，svd为30，eigh为10，eig为50）
矩阵维数大于30*30的用scipy的linalg包
矩阵维数小于30*30的用numpy的linalg包
求逆然后相乘最好用solve，如：dot(inv(a), b)=solve(a, b)
"""

import sys
import numpy as np
from numpy.linalg import multi_dot
import scipy as sp
from numpy import dot, diag, eye, mean, std, zeros, ones, sum, sqrt
from scipy.linalg import inv, solve, norm, eigh
from scipy.stats import t, f, chi2
import pandas as pd
from pprint import pprint
from sklearn.metrics import roc_curve, auc, accuracy_score
# from .LQ_SVM import kernel_function
from sklearn.metrics.pairwise import pairwise_kernels
from datetime import datetime
# from cvxopt import solvers, matrix
import itertools
from .LQ_SVM import *


__all__ = [ 'ROC_AUC', 'ROC_CI']





def ROC_AUC(Im, GroTru, CI=False, alpha=0.05):
    """
    计算ROC曲线的PF、PD、AUC

    Parameters
    ----------
    Im: 背景抑制后图像，可以是图像序列
    GroTru: 目标真实分布
    CI: 是否计算PF的置信区间
    alpha: 置信区间的显著性水平

    Returns
    -------

    """
    if Im.ndim is 2:
        m, n = Im.shape
        Pf, Pd, _ = roc_curve(np.reshape(GroTru, m * n), np.reshape(Im, m * n))
        AUC = auc(Pf, Pd)
        if AUC == 1:
            Pf = 10 ** (np.arange(-5, 0.5, 0.5))
            Pd = np.ones(11)
        if CI is True:
            PfL, PfU = ROC_CI(m * n, Pf, alpha=alpha)
            Pf = np.c_[PfL, Pf, PfU].T
            AUCL = auc(PfU, Pd)
            AUCU = auc(PfL, Pd)
            AUC = np.array([AUCL, AUC, AUCU])

    if Im.ndim is 3:
        m, n, p = Im.shape
        Pf = []
        Pd = []
        if CI is True:
            AUC = np.zeros((3, p))
        else:
            AUC = np.zeros(p)
        for i in range(p):
            Pf_i, Pd_i, _ = roc_curve(np.reshape(GroTru, m * n), np.reshape(Im[:, :, i], m * n))
            AUCi = auc(Pf_i, Pd_i)
            if AUCi == 1:
                Pf_i = 10 ** (np.arange(-5, 0.5, 0.5))
                Pd_i = np.ones(11)
            if CI is True:
                PfL_i, PfU_i = ROC_CI(m * n, Pf_i, alpha=alpha)
                Pf_i = np.c_[PfL_i, Pf_i, PfU_i].T
                AUCL = auc(PfU_i, Pd_i)
                AUCU = auc(PfL_i, Pd_i)
                AUCi = np.array([AUCL, AUCi, AUCU])
            Pf.append(Pf_i)
            Pd.append(Pd_i)
            if CI is True:
                AUC[:, i] = AUCi
            else:
                AUC[i] = AUCi
    return Pf, Pd, AUC


def ROC_CI(N, Vec_theta, alpha=0.05):
    """
    One-Dimensional Confidence-Interval Calculations
    Parameters
    ----------
    N
    Vec_theta
    alpha

    Returns
    -------
    theta_L
    theta_U
    """
    theta_L = np.zeros(Vec_theta.size)
    theta_U = np.zeros(Vec_theta.size)
    for i, theta in enumerate(Vec_theta):
        if theta != 0:
            alpha_2 = alpha / 2
        else:
            alpha_2 = alpha

        if N > 100 and theta > 0.1:
            d = N - 1
            sigma = sqrt(theta * (1 - theta))
            if theta == 0:
                theta_L[i] = 0
            else:
                theta_L[i] = theta - t.isf(alpha_2, df=d) * sigma / sqrt(N)
            theta_U[i] = theta + t.isf(alpha_2, df=d) * sigma / sqrt(N)
        elif N > 100 and theta < 0.1:
            if theta == 0:
                theta_L[i] = 0
            else:
                d_L = 2 * N * theta
                theta_L[i] = chi2.isf(1 - alpha_2, df=d_L) / (2 * N)
            d_U = 2 * (N * theta + 1)
            theta_U[i] = chi2.isf(alpha_2, df=d_U) / (2 * N)
        else:
            d1L = N - N * theta + 1
            d2L = N * theta
            if theta == 0:
                theta_L[i] = 0
            else:
                theta_L[i] = d2L / (d2L + d1L * f.isf(alpha_2, 2 * d1L, 2 * d2L))
            d1U = N * theta + 1
            d2U = N - N * theta
            theta_U[i] = d1U * f.isf(alpha_2, 2 * d1U, 2 * d2U) / (d2U + d1U * f.isf(alpha_2, 2 * d1U, 2 * d2U))

    # ensure increase
    for i in range(Vec_theta.size - 1):
        if theta_L[i + 1] < theta_L[i]:
            theta_L[i + 1] = theta_L[i]
        if theta_U[i + 1] < theta_U[i]:
            theta_U[i + 1] = theta_U[i]

    return theta_L, theta_U

