from sklearn.linear_model.base import LinearModel
from sklearn.base import RegressorMixin
from sklearn.utils import check_X_y
import numpy as np
import pandas as pd
from gurobipy import *
import time

class ConstrainedLinearRegression(LinearModel, RegressorMixin):

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True, nonnegative=False, tol=1e-15):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.nonnegative = nonnegative
        self.tol = tol

    def fit(self, X, y, min_coef=None, max_coef=None):
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'], y_numeric=True, multi_output=False)
        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize, copy=self.copy_X)
        
        t = time.time()
        m = Model("CSLR")
        coef = pd.Series(m.addVars(X.shape[1], lb = -1, ub = 1))
        intercept = m.addVar(lb = -1, ub = 1)
        obj = 0
        for i in range(X.shape[0]):
            obj_inner = 0
            for j in range(X.shape[1]):
                obj_inner += X[i][j]*coef[j]
            obj += (y[i] - obj_inner - intercept)*(y[i] - obj_inner - intercept)
        m.setObjective(obj, GRB.MINIMIZE)
        l2_norm = 0
        for i in range(X.shape[1]):
            l2_norm += coef[i]*coef[i]
        m.addConstr(l2_norm + intercept*intercept <= 1, "norm constraint")
        m.setParam('OutputFlag',0)
        m.optimize()
        print("elapse time:", time.time() - t)
        self.coef_ = [v.x for v in coef]
        self.intercept_ = intercept.x
        return self    
    
from sklearn.datasets import load_boston
#from sklearn.linear_model import LinearRegression
X, y = load_boston(return_X_y=True)
model = ConstrainedLinearRegression()
model.fit(X, y, -1*np.ones(13), np.ones(13))
print(model.intercept_)
print(model.coef_)