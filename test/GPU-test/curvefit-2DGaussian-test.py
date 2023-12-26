import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pygpufit.gpufit as gf
import torch
from torchimize.functions import lsq_lma
from torchimize.functions.jacobian import jacobian_approx_loop
import time

class scipy_curve_fit_tester:
    def __init__(self, num_points=100000, repeat=5, tolerance=1e-6, max_num_iter=50):

        self.num_points = int(num_points)
        self.true_param = np.array((4, self.num_points/2, self.num_points/20, 1), dtype=np.float32) # amp, xcenter, sigma, offset  

        self.init_param = [3, self.num_points/1.8, self.num_points/40, 0]
        self.repeat = int(repeat)
        self.tolerance = tolerance
        self.max_num_iter = int(max_num_iter)

    def generate_data(self):

        self.xdata = np.arange(self.num_points, dtype=np.float32)

        self.ydata = self.gaussian_1d(self.xdata, *self.true_param)
        rng = np.random.default_rng(seed=0)
        self.ydata += (rng.random(self.xdata.shape)-0.5) * self.ydata / 4 # add nosie
        self.ydata = np.array(self.ydata, dtype=np.float32)

    def gaussian_1d(self, x, amp, xcenter, sigma, offset):

        return amp * np.exp(- (x - xcenter)**2 / (2 * sigma**2)) + offset
    
    def curve_fit_test(self, print_time=True):

        self.generate_data()

        elapsed_time_list = []
        for _ in range(self.repeat):
            t0 = time.time()
            popt, pcov = scipy.optimize.curve_fit(self.gaussian_1d, self.xdata, self.ydata, p0=self.init_param, maxfev=self.max_num_iter, ftol=self.tolerance)
            elapsed_time_list.append(time.time() - t0)

        if print_time:
            print('Scipy elapsed times: ' + ', '.join(['{:.3f}'.format(i) for i in elapsed_time_list]) + ' s.')
            print('Scipy average time: {:.3f} s.'.format(np.mean(elapsed_time_list)))

        return (popt, self.true_param, self.init_param, self.xdata, self.ydata, self.gaussian_1d(self.xdata, *popt))
    
class gpufit_curve_fit_tester(scipy_curve_fit_tester):
    def __init__(self, num_points=None, repeat=None, tolerance=None, max_num_iter=None):

        kwargs = {}
        if num_points is not None:
            kwargs['num_points'] = num_points
        if repeat is not None:
            kwargs['repeat'] = repeat
        if tolerance is not None:
            kwargs['tolerance'] = tolerance
        if max_num_iter is not None:
            kwargs['max_num_iter'] = max_num_iter

        super().__init__(**kwargs)

        self.init_param = np.array([self.init_param], dtype=np.float32) # change format and make it a 2D array

    def generate_data(self):

        super().generate_data()

        self.ydata = np.array([self.ydata], dtype=np.float32) # change format and make it a 2D array

    def curve_fit_test(self, print_time=True):

        self.generate_data()

        exec_time_list = []
        elapsed_time_list = []
        for _ in range(self.repeat):
            t0 = time.time()
            fitted_param, states, chi_squares, number_iterations, execution_time = gf.fit(data = self.ydata, 
                                                                                        weights = None, 
                                                                                        model_id = gf.ModelID.GAUSS_1D,
                                                                                        initial_parameters = self.init_param, 
                                                                                        tolerance = self.tolerance,
                                                                                        max_number_iterations = self.max_num_iter, 
                                                                                        parameters_to_fit = None, 
                                                                                        estimator_id = None, 
                                                                                        user_info = None)
            elapsed_time_list.append(time.time() - t0)
            exec_time_list.append(execution_time)
        
        if print_time:
            print('gpufit elapsed times: ' + ', '.join(['{:.3f}'.format(i) for i in elapsed_time_list]) + ' s.')
            print('gpufit average time: {:.3f} s.'.format(np.mean(elapsed_time_list)))

        return (fitted_param[0], self.true_param, self.init_param[0], self.xdata, self.ydata[0], self.gaussian_1d(self.xdata, *fitted_param[0]))

class torchimize_curve_fit_tester(scipy_curve_fit_tester):
    def __init__(self, num_points=None, repeat=None, tolerance=None, max_num_iter=None):

        kwargs = {}
        if num_points is not None:
            kwargs['num_points'] = num_points
        if repeat is not None:
            kwargs['repeat'] = repeat
        if tolerance is not None:
            kwargs['tolerance'] = tolerance
        if max_num_iter is not None:
            kwargs['max_num_iter'] = max_num_iter

        super().__init__(**kwargs)

        self.init_param = torch.tensor(self.init_param, dtype=torch.float32, device='cuda')
        self.max_num_iter = torch.tensor(self.max_num_iter, dtype=torch.int32, device='cuda')
        self.tolerance = torch.tensor(self.tolerance, dtype=torch.float32, device='cuda')

    def generate_data(self):

        super().generate_data()

        self.xdata = torch.tensor(self.xdata, dtype=torch.float32, device='cuda')
        self.ydata = torch.tensor(self.ydata, dtype=torch.float32, device='cuda')

    def gaussian_1d_torch(self, x, amp, xcenter, sigma, offset):
            
        return amp * torch.exp(- (x - xcenter)**2 / (2 * sigma**2)) + offset

    def cost_func(self, param, x, y):

        return (y - self.gaussian_1d_torch(x, *param))**2
    
    def curve_fit_test(self, print_time=True):

        self.generate_data()

        # traced_cost_func = torch.jit.trace(self.cost_func, (self.init_param, self.xdata, self.ydata))

        jac_func = lambda p, xdata, ydata: jacobian_approx_loop(p, f=self.cost_func, dp=1e-6, args=(xdata, ydata))

        elapsed_time_list = []
        for _ in range(self.repeat):
            t0 = time.time()

            # returns a list of tensors consisting of fitted parameters found in each iteration
            fitted_param = lsq_lma(self.init_param, function=self.cost_func, jac_function=jac_func, 
                                    args=(self.xdata, self.ydata), max_iter=self.max_num_iter, 
                                    ftol=self.tolerance, gtol=self.tolerance, ptol=self.tolerance)
            
            torch.cuda.synchronize()
            elapsed_time_list.append(time.time() - t0)

        if print_time:
            print('PyTorch elapsed times: ' + ', '.join(['{:.3f}'.format(i) for i in elapsed_time_list]) + ' s.')
            print('PyTorch average time: {:.3f} s.'.format(np.mean(elapsed_time_list)))
    
        return (fitted_param[-1].cpu().numpy(), self.true_param, self.init_param.cpu().numpy(), 
                self.xdata.cpu().numpy(), self.ydata.cpu().numpy(), 
                self.gaussian_1d(self.xdata.cpu().numpy(), *fitted_param[-1].cpu().numpy()))


# num_points = 100
scipy_test = scipy_curve_fit_tester()
fitted_param, true_param, init_param, xdata, ydata, scipy_fitted_ydata = scipy_test.curve_fit_test(print_time=True)

gpufit_test = gpufit_curve_fit_tester()
fitted_param, true_param, init_param, xdata, ydata, gpufit_fitted_ydata = gpufit_test.curve_fit_test(print_time=True)

# slow and less accurate for large dataset, need to increase max_num_iter and tolerance
torchimize_test = torchimize_curve_fit_tester(max_num_iter=100, tolerance=1e-8)
fitted_param, true_param, init_param, xdata, ydata, torchimize_fitted_ydata = torchimize_test.curve_fit_test(print_time=True)

plt.plot(xdata, ydata, 'o')
plt.plot(xdata, scipy_fitted_ydata, '--', label='Scipy fit')
plt.plot(xdata, gpufit_fitted_ydata, '--', label='Gpufit fit')
plt.plot(xdata, torchimize_fitted_ydata, '--', label='PyTorch fit')
plt.legend()
plt.grid()
plt.show()