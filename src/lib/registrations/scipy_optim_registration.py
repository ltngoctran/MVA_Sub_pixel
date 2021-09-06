import numpy as np
from scipy.optimize import minimize
from ..tools.utils import fourier_shift

def scipy_registration_method(original,target):
    def obj_fun(t,original,target):
        F = fourier_shift(original, t)
        return np.sum((F-target)**2)
    bnds = [(0, 1),(0, 1)]
    xinit = np.random.rand(2)
    res = minimize(fun=obj_fun, args=(original, target), x0=xinit, method='SLSQP', bounds=bnds,  tol=1e-9)
    return res.x