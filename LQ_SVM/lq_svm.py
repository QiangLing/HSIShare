"""
c++ lqsvm 模块封装
kernel: rbf linear poly sigmoid
CSVC, CSR_TD, and CSRBBH_TD: X is a ndarray whose each row is a training sample, the order of these samples is [target samples, background samples], Label is the label of these samples, +1 for target class, -1 for background class.
"""

from . import lqsvm
import numpy as np

__all__ = ['CSR_AD', 'CSR_AD2']


class CSR_AD(lqsvm.CSR_AD):
    def __init__(self, kernel='rbf', solver='SMO', p=0, nu=0.2, degree=3, gamma=0.02, coef0=0.0,
                 tol=1e-5, shrinking=False, cache_size=200, verbose=False):

        super(CSR_AD, self).__init__(kernel, solver, degree, gamma, coef0, p, cache_size,
                                     tol, nu, shrinking, verbose)
        self.solver = solver
    def decision_function(self, X, y, sample_weight=None):
        if sample_weight is None:
            if self.solver == 'SMO':
                SW = np.ones(X.shape[0])
            elif self.solver == 'DCD':
                SW = np.inf*np.ones(X.shape[0])
        else:
            SW = sample_weight
        dec = super(CSR_AD, self).decision_function(X, y, SW)
        self.alpha = np.zeros(self.total_SV)
        super(CSR_AD, self).get_sv_coef(self.alpha)
        self.indices = np.zeros(self.total_SV, dtype=int)
        super(CSR_AD, self).get_sv_indices(self.indices)
        return dec

    def decision_function1(self, X, y, sample_weight=None):
        if sample_weight is None:
            SW = np.ones(X.shape[0])
        else:
            SW = sample_weight
        dec = super(CSR_AD, self).decision_function1(X, y, SW)
        self.alpha = np.zeros(self.total_SV)
        super(CSR_AD, self).get_sv_coef(self.alpha)
        self.indices = np.zeros(self.total_SV, dtype=int)
        super(CSR_AD, self).get_sv_indices(self.indices)
        return dec


class CSR_AD2(lqsvm.CSR_AD2):
    def __init__(self, kernel='rbf', solver='SMO', p=0, nu=0.2, degree=3, gamma=0.02, coef0=0.0,
                 tol=1e-5, shrinking=False, cache_size=200, verbose=False):

        super(CSR_AD2, self).__init__(kernel, solver, degree, gamma, coef0, p, cache_size,
                                      tol, nu, shrinking, verbose)

    def decision_function(self, X, y, sample_weight=None):
        if sample_weight is None:
            SW = np.ones(X.shape[0])
        else:
            SW = sample_weight
        dec = super(CSR_AD2, self).decision_function(X, y, SW)
        self.alpha0 = np.zeros(self.total_SV0)
        super(CSR_AD2, self).get_sv_coef0(self.alpha0)
        self.indices0 = np.zeros(self.total_SV0, dtype=int)
        super(CSR_AD2, self).get_sv_indices0(self.indices0)
        self.alpha = np.zeros(self.total_SV)
        super(CSR_AD2, self).get_sv_coef(self.alpha)
        self.indices = np.zeros(self.total_SV, dtype=int)
        super(CSR_AD2, self).get_sv_indices(self.indices)
        return dec
