"""
测试单个高光谱异常检测算法
"""
from LQ import *
import scipy.io as spio
from time import time

rcParams['savefig.format'] = 'png'
#

HSI, GroTru, GoodBands = load_SHARE2012(ROI='AVON_AM')
OriginalIm = HSI[:, :, [52, 33, 15]]
# GoodBands = GoodBands[::3]


HSI = HSI[:, :, GoodBands]
HSI = HSINormalization(HSI)
# endregion
Windows = (3, 9)
Kernel = 'rbf'   # 'linear'  'rbf'
tol = 1e-5  # 1e-4  1e-5
gamma = 0.2
nu = 1e-5

num = 1
runtime = np.zeros(num)
AUC = np.zeros(num)
# Gamma = np.linspace(0.1, 0.1, num=num)   # 0.06 0.09+0.04 5.5+3 7
Nu = np.linspace(0.01, 0.3, num=num)
for i, nu in enumerate(Nu):
    t1 = time()
    Imi = HSI_CSR_AD2(HSI, Windows=Windows, Kernel=Kernel, gamma=gamma, nu=nu, tol=tol)  # 0.2,0.2 0.3,0.08  3.0,0.025
    runtime[i] = time() - t1
    Pfi, Pdi, AUC[i] = ROC_AUC(Imi, GroTru)
    if np.argmax(AUC) == i:
        Pf = Pfi
        Pd = Pdi
        Im = Imi

# spio.savemat('Im', {'Im': Im})
print(runtime)
print(AUC)
rcParams['lines.markersize'] = 0.0
plt.figure(1)
ImshowNormalization(OriginalIm, WindowTitle='原始图像')
plt.figure(2)
ImshowNormalization(Im, Cmap='jet', ColorBarOn=False)
plt.figure(3)
ImshowNormalization(GroTru)
plt.figure(4)
plt.plot(Pf, Pd, linewidth=2)
# plt.plot(Pf1, Pd1, linewidth=2)
plt.xscale('log')
plt.xlim([1e-5, 1.0])
# plt.ylim([0.0, 1.0])
XLabel = '虚警概率'
YLabel = '检测概率'
PlotNormalization(XLabel, YLabel)
plt.show()
