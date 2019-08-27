"""
自定义包LQ（标准数据分析包）

数据分析标准包，包含常用的数据分析模块
import模块会导入模块里定义的函数、变量以及模块里导入的模块
提供规范化绘图参数、规范化绘图函数等（如PlotNormalization）
提供常用的数据处理函数
-----------------------------重要（算法运行速度提升）---------------------------------------
numpy的linalg包是轻量级的，针对小矩阵，scipy的linalg包是重量级的，针对大矩阵：
（不同函数维数不一样，inv为15，svd为30，eigh为10，eig为50）
矩阵维数大于30*30的用scipy的linalg包
矩阵维数小于30*30的用numpy的linalg包
求逆然后相乘最好用solve，如：dot(inv(a), b)=solve(a, b)
"""

from .DataAnalysis import *
from .FigureNormalization import *
from .HSIAnalysis import *
# from .SpectralAnalysis import *
from .LQ_SVM import *
from .Datasets import *
