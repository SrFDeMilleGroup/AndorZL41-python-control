import numpy as np
from scipy.optimize import leastsq

from pytorch_curve_fit import gaussian_2d_fit_lma

def gaussian_2d(self, amp, x_mean, y_mean, x_sigma, y_sigma, offset):

        return lambda x, y: amp*np.exp(-0.5*((x-x_mean)/x_sigma)**2-0.5*((y-y_mean)/y_sigma)**2) + offset

def gaussian_2d_fit(data: np.ndarray, max_iter: int = 50, tol: float = 1e-8, GPU_fit: bool = False):

    """
    2D gaussian fitting of a 2D array

    arguments:
    ---------------------------------
    data -> np.ndarray: 2D numpy array to fit
    max_iter -> int: max number of iterations for fitting
    tol -> float: tolerance for fitting
    GPU_fit -> bool: boolean, use GPU fitting or not. If not, use scipy leastsq fitting on CPU

    returns:
    ---------------------------------
    popt -> np.ndarray: 1D numpy array of fitted parameters, in order of [amplitude, x_mean, y_mean, x_sigma, y_sigma, offset]
    """
    
    # calculate moments for initial guess
    # codes adapted from https://scipy-cookbook.readthedocs.io/items/FittingData.html
    # generally a 2D gaussian fit can have 7 params, 6 of them are implemented here (the excluded one is an angle)
    total = np.sum(data)
    X, Y = np.indices(data.shape)
    x_mean = np.sum(X*data)/total
    x_mean = np.clip(x_mean, 0, data.shape[0]-1) # coerce x_mean to data shape
    y_mean = np.sum(Y*data)/total
    y_mean = np.clip(y_mean, 0, data.shape[1]-1) # coerce y_mean to data shape
    col = data[:, int(y_mean)]
    x_sigma = np.sqrt(np.abs((np.arange(col.size)-x_mean)**2*col).sum()/col.sum())
    row = data[int(x_mean), :]
    y_sigma = np.sqrt(np.abs((np.arange(row.size)-y_mean)**2*row).sum()/row.sum())
    offset = (data[0, :].sum()+data[-1, :].sum()+data[:, 0].sum()+data[:, -1].sum())/np.sum(data.shape)/2
    amp = data.max() - offset

    init_param = np.array([amp, x_mean, y_mean, x_sigma, y_sigma, offset], dtype=np.float32)

    if GPU_fit:
        # use GPU fitting
        popt, i, success = gaussian_2d_fit_lma(init_param, X, Y, data, max_iter=max_iter, meth='lev', ftol=tol, gtol=tol, ptol=tol)

    else:
        errorfunction = lambda p: np.ravel(gaussian_2d(*p)(X, Y) - data.astype(np.float32))
        popt, pcov, infodict, msg, success = leastsq(errorfunction, init_param, maxfev=max_iter, ftol=tol, full_output=True)

    return popt