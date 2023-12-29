import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pygpufit.gpufit as gf
import torch
from torchimize.functions import lsq_lma
from torchimize.functions.jacobian import jacobian_approx_loop
from pytorch_curve_fit import gaussian_1d_fit_lma
import time

class scipy_curve_fit_tester:
    def __init__(self, array_size=1000000, noise_level=0.25, repeat=5, tolerance=1e-6, max_num_iter=50):

        self.array_size = int(array_size)
        self.noise_level = noise_level
        self.true_param = np.array((4, self.array_size/2, self.array_size/20, 1), dtype=np.float32) # amp, xcenter, sigma, offset  

        self.init_param = np.array([3, self.array_size/1.8, self.array_size/40, 0], dtype=np.float32)
        self.repeat = int(repeat)
        self.tolerance = tolerance
        self.max_num_iter = int(max_num_iter)

    def generate_data(self):

        xdata = np.arange(self.array_size, dtype=np.float32)

        ydata = self.gaussian_1d(xdata, *self.true_param)
        rng = np.random.default_rng(seed=0)
        ydata += (rng.random(xdata.shape)-0.5) * ydata * self.noise_level # add nosie
        ydata = np.array(ydata, dtype=np.float32)

        return (xdata, ydata)

    def gaussian_1d(self, x, amp, xcenter, sigma, offset):

        return amp * np.exp(- (x - xcenter)**2 / (2 * sigma**2)) + offset
    
    def curve_fit_test(self, print_time=True):

        xdata, ydata = self.generate_data()

        elapsed_time_list = []
        for _ in range(self.repeat):
            t0 = time.time()
            popt, pcov, infodict, msg, success = scipy.optimize.curve_fit(self.gaussian_1d, xdata, ydata, 
                                                  p0=self.init_param, 
                                                  maxfev=self.max_num_iter, 
                                                  ftol=self.tolerance,
                                                  full_output=True)
            
            elapsed_time_list.append(time.time() - t0)
            assert success in [1,2,3,4], 'Scipy: Fit failed.\n ' + msg

        if print_time:
            print('Scipy elapsed times: ' + ', '.join(['{:.3f}'.format(i) for i in elapsed_time_list]) + ' s.')
            print('Scipy average time: {:.3f} s.'.format(np.mean(elapsed_time_list)))

        return (popt, self.true_param, self.init_param, xdata, ydata, self.gaussian_1d(xdata, *popt))
    
class gpufit_curve_fit_tester(scipy_curve_fit_tester):

    def curve_fit_test(self, print_time=True):

        xdata, ydata = self.generate_data()
        ydata = np.array([ydata], dtype=np.float32) # change format and make it a 2D array

        self.init_param = np.array([self.init_param], dtype=np.float32) # change format and make it a 2D array
        exec_time_list = []
        elapsed_time_list = []
        for _ in range(self.repeat):
            t0 = time.time()
            fitted_param, states, chi_squares, number_iterations, execution_time = gf.fit(data = ydata, 
                                                                                        weights = None, 
                                                                                        model_id = gf.ModelID.GAUSS_1D,
                                                                                        initial_parameters = self.init_param, 
                                                                                        tolerance = self.tolerance,
                                                                                        max_number_iterations = self.max_num_iter, 
                                                                                        parameters_to_fit = None, 
                                                                                        estimator_id = gf.EstimatorID.LSE, 
                                                                                        user_info = None)
            elapsed_time_list.append(time.time() - t0)
            exec_time_list.append(execution_time)

            assert states[0] == gf.Status.Ok, 'gpufit: Fit failed.'
        
        if print_time:
            print('gpufit elapsed times: ' + ', '.join(['{:.3f}'.format(i) for i in elapsed_time_list]) + ' s.')
            print('gpufit average time: {:.3f} s.'.format(np.mean(elapsed_time_list)))

        return (fitted_param[-1], self.true_param, self.init_param[0], xdata, ydata[0], self.gaussian_1d(xdata, *fitted_param[0]))

class torchimize_curve_fit_tester(scipy_curve_fit_tester):

    def gaussian_1d_torch(self, x, amp, xcenter, sigma, offset):
            
        return amp * torch.exp(- (x - xcenter)**2 / (2 * sigma**2)) + offset
    
    def curve_fit_test(self, print_time=True):

        xdata, ydata = self.generate_data()
        xdata = torch.tensor(xdata, dtype=torch.float32, device='cuda:0')
        ydata = torch.tensor(ydata, dtype=torch.float32, device='cuda:0')
        self.init_param = torch.tensor(self.init_param, dtype=torch.float32, device='cuda:0')

        # traced_cost_func = torch.jit.trace(self.cost_func, (self.init_param, xdata, ydata))

        cost_func = lambda p, xdata, ydata: self.gaussian_1d_torch(xdata, *p) - ydata
        jac_func = lambda p, xdata, ydata: jacobian_approx_loop(p, f=cost_func, dp=1e-6, args=(xdata, ydata))

        elapsed_time_list = []
        for _ in range(self.repeat):
            t0 = time.time()

            # returns a list of tensors consisting of fitted parameters found in each iteration
            fitted_param = lsq_lma(self.init_param, function=cost_func, jac_function=jac_func, 
                                    args=(xdata, ydata), max_iter=self.max_num_iter, 
                                    meth='lev',
                                    ftol=self.tolerance, gtol=self.tolerance, ptol=self.tolerance)
            
            torch.cuda.synchronize()
            elapsed_time_list.append(time.time() - t0)

        if print_time:
            print('PyTorch elapsed times: ' + ', '.join(['{:.3f}'.format(i) for i in elapsed_time_list]) + ' s.')
            print('PyTorch average time: {:.3f} s.'.format(np.mean(elapsed_time_list)))
    
        return (fitted_param[-1].cpu().numpy(), self.true_param, self.init_param.cpu().numpy(), 
                xdata.cpu().numpy(), ydata.cpu().numpy(), 
                self.gaussian_1d(xdata.cpu().numpy(), *fitted_param[-1].cpu().numpy()))


class modified_torch_curve_fit_tester(torchimize_curve_fit_tester):
    
    def curve_fit_test(self, print_time=True):

        xdata, ydata = self.generate_data()

        elapsed_time_list = []
        for _ in range(self.repeat):
            t0 = time.time()

            # returns a list of tensors consisting of fitted parameters found in each iteration
            fitted_param, i, success = gaussian_1d_fit_lma(self.init_param, xdata, ydata, 
                                               max_iter=self.max_num_iter, 
                                                meth='lev',
                                                ftol=self.tolerance, gtol=self.tolerance, ptol=self.tolerance)
            
            torch.cuda.synchronize()
            elapsed_time_list.append(time.time() - t0)

        if print_time:
            print('my code elapsed times: ' + ', '.join(['{:.3f}'.format(i) for i in elapsed_time_list]) + ' s.')
            print('my code average time: {:.3f} s.'.format(np.mean(elapsed_time_list)))

        return (fitted_param, self.true_param, self.init_param, 
                xdata, ydata, self.gaussian_1d(xdata, *fitted_param))

# array_size = 100
scipy_test = scipy_curve_fit_tester()
fitted_param, true_param, init_param, xdata, ydata, scipy_fitted_ydata = scipy_test.curve_fit_test(print_time=True)

gpufit_test = gpufit_curve_fit_tester()
fitted_param, true_param, init_param, xdata, ydata, gpufit_fitted_ydata = gpufit_test.curve_fit_test(print_time=True)

torchimize_test = torchimize_curve_fit_tester(tolerance=1e-8)
fitted_param, true_param, init_param, xdata, ydata, torchimize_fitted_ydata = torchimize_test.curve_fit_test(print_time=True)

modified_torch_test = modified_torch_curve_fit_tester(tolerance=1e-8)
fitted_param, true_param, init_param, xdata, ydata, modified_torch_fitted_ydata = modified_torch_test.curve_fit_test(print_time=True)

plt.plot(xdata, ydata, 'o')
plt.plot(xdata, scipy_fitted_ydata, '--', label='Scipy fit')
plt.plot(xdata, gpufit_fitted_ydata, '--', label='Gpufit fit')
plt.plot(xdata, torchimize_fitted_ydata, '--', label='PyTorch fit')
plt.plot(xdata, modified_torch_fitted_ydata, '--', label='My code fit')
plt.legend()
plt.grid()
plt.show()