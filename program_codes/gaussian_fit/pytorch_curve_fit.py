"""
Adapted from https://github.com/hahnec/torchimize/blob/master/torchimize/functions/single/lma_fun_single.py

This python script defines PyTorch-based GPU-accelerated Gaussain curve fit functions. 
These functions are adapted from the torchimize package, as mentioned above. 
Curve fit is done by minimizing the sum of squared residuals (least squares optimization) using the Levenberg-Marquardt algorithm.
We modify the torchimize package by explicitely implementing the cost function and Jacobian matrix, 
to reduce the number of function calls and therefore evaluation time.

A similar CuPy version of the curve fit functions is also developed and tested, but not included here, 
because it seems slower and less accurate (CuPy 12.3 with CUDA 12.3 vs torch 2.1.2 with CUDA 12.1). 

Below is a good introduction to the Levenberg-Marquardt algorithm:
'The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems' by Henri. P. Gavin
https://people.duke.edu/~hpgavin/ExperimentalSystems/lm.pdf

"""

import numpy as np
import torch
import logging

__all__ = ['gaussian_1d_fit_lma', 'gaussian_2d_fit_lma']

def gaussian_1d_fit_lma(p: np.ndarray, xdata: np.ndarray, ydata: np.ndarray, 
                        tau: float = 1e-3, rho1: float = .25, rho2: float = .75, 
                        beta: float = 2, gamma: float = 3,
                        meth: str = 'lev', max_iter: int = 100,
                        ftol: float = 1e-8, ptol: float = 1e-8, gtol: float = 1e-8, 
                        ) -> tuple[np.ndarray, int, bool]:
    
    """
    Levenberg-Marquardt implementation for least-squares fitting of 1D gaussian functions

    arguments:
    --------------------------------------------------
    p -> np.ndarray: initial guess of fitting parameters, in order of [amplitude, x_mean, x_sigma, offset]
    xdata -> np.ndarray: x values of data
    ydata -> np.ndarray: y values of data
    tau -> float: factor to initialize damping parameter
    rho1 -> float: first gain factor threshold for damping parameter adjustment for Marquardt method
    rho2 -> float: second gain factor threshold for damping parameter adjustment for Marquardt method
    beta -> float: multiplier for damping parameter adjustment for Marquardt method
    gamma -> float: divisor for damping parameter adjustment for Marquardt method
    meth -> str: method which is by default 'lev' for Levenberg method and 'marq' for Marquardt method
    max_iter -> int: maximum number of iterations
    ftol -> float: relative change in cost function as stop condition
    ptol -> float: relative change in independant variables as stop condition
    gtol -> float: maximum gradient tolerance as stop condition

    return:
    --------------------------------------------------
    p -> np.ndarray: fitted parameters, in order of [amplitude, x_mean, x_sigma, offset]
    i -> int: number of iterations
    success -> bool: whether the fit is successful
    """
        
    if meth not in ['lev', 'marq']:
        logging.warning(f'GPU curve fit: invalid method {meth}. Use method lev instead.')
        meth = 'lev'

    assert len(p) == 4, 'Gaussian 1D fit: p must be a 4-element tensor in order of [amplitude, x_mean, x_sigma, offset].'
    assert rho2 >= rho1, 'Gaussian 1D fit: rho2 must be greater than rho1.'

    cost_func = lambda p, x, y: p[0] * torch.exp(- (x - p[1])**2 / (2 * p[2]**2)) + p[3] - y

    p = torch.tensor(p, dtype=torch.float32, device='cuda:0')
    xdata = torch.tensor(xdata, dtype=torch.float32, device='cuda:0')
    ydata = torch.tensor(ydata, dtype=torch.float32, device='cuda:0')

    f = cost_func(p, xdata, ydata)
    j = torch.ones((xdata.shape[0], 4), dtype=torch.float32, device='cuda:0')
    j[:, 0] = (f+ydata-p[3])/p[0]
    j[:, 1] = (f+ydata-p[3])*(xdata-p[1])/p[2]**2
    j[:, 2] = j[:, 1]*(xdata-p[1])/p[2]

    g = torch.matmul(j.T, f)
    H = torch.matmul(j.T, j)
    u = tau * torch.max(torch.diag(H))
    v = 2
    i = 0
    success = False
    D = torch.eye(p.shape[0], dtype=torch.float32, device='cuda:0')
    while i < max_iter:
        D2 = D*torch.max(torch.maximum(H.diagonal(), D.diagonal())) if meth == 'marq' else D.clone()
        h = -torch.linalg.lstsq(H+u*D2, g, rcond=None, driver=None)[0]
        f_h = cost_func(p+h, xdata, ydata)
        rho_denom = torch.matmul(h, u*h-g)
        rho_nom = torch.matmul(f, f) - torch.matmul(f_h, f_h)
        rho = rho_nom / rho_denom if rho_denom > 0 else torch.inf if rho_nom > 0 else -torch.inf
        if rho > 0:
            p = p + h
            j[:, 0] = (f_h+ydata-p[3])/p[0]
            j[:, 1] = (f_h+ydata-p[3])*(xdata-p[1])/p[2]**2
            j[:, 2] = j[:, 1]*(xdata-p[1])/p[2]
            
            g = torch.matmul(j.T, f_h)
            H = torch.matmul(j.T, j)
        f_prev = f.clone()
        f = f_h.clone() if rho > 0 else f

        if meth == 'lev':
            if rho > 0:
                u = u * torch.max(torch.tensor([1/3, 1-(2*rho-1)**3]))
                v = 2
            else:
                u = u*v
                v = 2*v
        elif meth == 'marq':
            if rho < rho1:
                u = u*beta
            elif rho > rho2:
                u = u/gamma
            else:
                u = u
        else:
            raise ValueError('GPU curve fit: invalid meth argument.')

        # stop conditions
        gcon = torch.max(torch.abs(g)) < gtol
        pcon = (h**2).sum()**.5 < ptol*(ptol + (p**2).sum()**.5)
        fcon = ((f_prev-f)**2).sum() < ((ftol*f)**2).sum() if rho > 0 else False
        if gcon or pcon or fcon:
            success = True
            break

        i += 1

    return (p.cpu().numpy(), i, success)


def gaussian_2d_fit_lma(p: np.ndarray, xdata: np.ndarray, ydata: np.ndarray, zdata: np.ndarray,
                        tau: float = 1e-3, rho1: float = .25, rho2: float = .75, 
                        beta: float = 2, gamma: float = 3,
                        meth: str = 'lev', max_iter: int = 100,
                        ftol: float = 1e-8, ptol: float = 1e-8, gtol: float = 1e-8, 
                        ) -> tuple[np.ndarray, int, bool]:
    
    """
    Levenberg-Marquardt implementation for least-squares fitting of 2D gaussian functions

    arguments:
    --------------------------------------------------
    p -> np.ndarray: initial guess of fitting parameters, in order of [amplitude, x_mean, y_mean, x_sigma, y_sigma, offset]
    xdata -> np.ndarray: x values of data, 2D array
    ydata -> np.ndarray: y values of data, 2D array
    zdata -> np.ndarray: z values of data, 2D array
    tau -> float: factor to initialize damping parameter
    rho1 -> float: first gain factor threshold for damping parameter adjustment for Marquardt method
    rho2 -> float: second gain factor threshold for damping parameter adjustment for Marquardt method
    beta -> float: multiplier for damping parameter adjustment for Marquardt method
    gamma -> float: divisor for damping parameter adjustment for Marquardt method
    meth -> str: method which is by default 'lev' for Levenberg method and 'marq' for Marquardt method
    max_iter -> int: maximum number of iterations
    ftol -> float: relative change in cost function as stop condition
    ptol -> float: relative change in independant variables as stop condition
    gtol -> float: maximum gradient tolerance as stop condition

    return:
    --------------------------------------------------
    p -> np.ndarray: fitted parameters, in order of [amplitude, x_mean, y_mean, x_sigma, y_sigma, offset]
    i -> int: number of iterations
    success -> bool: whether the fit is successful
    """
        
    if meth not in ['lev', 'marq']:
        logging.warning(f'GPU curve fit: invalid method {meth}. Use method lev instead.')
        meth = 'lev'

    assert len(p) == 6, 'Gaussian 2D fit: p must be a 6-element tensor in order of [amplitude, x_mean, y_mean, x_sigma, y_sigma, offset].'
    assert rho2 >= rho1, 'Gaussian 2D fit: rho2 must be greater than rho1.'

    cost_func = lambda p, x, y, z: p[0] * torch.exp(- (x-p[1])**2 / (2*p[3]**2) - (y-p[2])**2 / (2*p[4]**2)) + p[5] - z

    p = torch.tensor(p, dtype=torch.float32, device='cuda:0')
    xdata_flat = torch.tensor(xdata.ravel(), dtype=torch.float32, device='cuda:0')
    ydata_flat = torch.tensor(ydata.ravel(), dtype=torch.float32, device='cuda:0')
    zdata_flat = torch.tensor(zdata.ravel(), dtype=torch.float32, device='cuda:0')

    f = cost_func(p, xdata_flat, ydata_flat, zdata_flat)
    j = torch.ones((xdata_flat.shape[0], 6), dtype=torch.float32, device='cuda:0')
    j[:, 0] = (f+zdata_flat-p[5])/p[0]
    j[:, 1] = (f+zdata_flat-p[5])*(xdata_flat-p[1])/p[3]**2
    j[:, 2] = (f+zdata_flat-p[5])*(ydata_flat-p[2])/p[4]**2
    j[:, 3] = j[:, 1]*(xdata_flat-p[1])/p[3]
    j[:, 4] = j[:, 2]*(ydata_flat-p[2])/p[4]

    g = torch.matmul(j.T, f)
    H = torch.matmul(j.T, j)
    u = tau * torch.max(torch.diag(H))
    v = 2
    i = 0
    success = False
    D = torch.eye(p.shape[0], dtype=torch.float32, device='cuda:0')
    while i < max_iter:
        D2 = D*torch.max(torch.maximum(H.diagonal(), D.diagonal())) if meth == 'marq' else D.clone()
        h = -torch.linalg.lstsq(H+u*D2, g, rcond=None, driver=None)[0]
        f_h = cost_func(p+h, xdata_flat, ydata_flat, zdata_flat)
        rho_denom = torch.matmul(h, u*h-g)
        rho_nom = torch.matmul(f, f) - torch.matmul(f_h, f_h)
        rho = rho_nom / rho_denom if rho_denom > 0 else torch.inf if rho_nom > 0 else -torch.inf
        if rho > 0:
            p = p + h
            j[:, 0] = (f_h+zdata_flat-p[5])/p[0]
            j[:, 1] = (f_h+zdata_flat-p[5])*(xdata_flat-p[1])/p[3]**2
            j[:, 2] = (f_h+zdata_flat-p[5])*(ydata_flat-p[2])/p[4]**2
            j[:, 3] = j[:, 1]*(xdata_flat-p[1])/p[3]
            j[:, 4] = j[:, 2]*(ydata_flat-p[2])/p[4]
            
            g = torch.matmul(j.T, f_h)
            H = torch.matmul(j.T, j)
        f_prev = f.clone()
        f = f_h.clone() if rho > 0 else f

        if meth == 'lev':
            if rho > 0:
                u = u * torch.max(torch.tensor([1/3, 1-(2*rho-1)**3]))
                v = 2
            else:
                u = u*v
                v = 2*v
        elif meth == 'marq':
            if rho < rho1:
                u = u*beta
            elif rho > rho2:
                u = u/gamma
            else:
                u = u
        else:
            raise ValueError('GPU curve fit: invalid meth argument.')

        # stop conditions
        gcon = torch.max(torch.abs(g)) < gtol
        pcon = (h**2).sum()**.5 < ptol*(ptol + (p**2).sum()**.5)
        fcon = ((f_prev-f)**2).sum() < ((ftol*f)**2).sum() if rho > 0 else False
        if gcon or pcon or fcon:
            success = True
            break

        i += 1

    return (p.cpu().numpy(), i, success)