import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pygpufit.gpufit as gf
import torch
from torchimize.functions import lsq_lma
from torchimize.functions.jacobian import jacobian_approx_loop
from pytorch_curve_fit import gaussian_2d_fit_lma
import time

class scipy_curve_fit_tester:
    def __init__(self, array_size_x=1000, array_size_y=1000, repeat=5, tolerance=1e-6, max_num_iter=50):

        self.array_size_x = int(array_size_x)
        self.array_size_y = int(array_size_y)
        # amp, x_mean, y_mean, x_sigma, y_sigma, offset  
        self.true_param = np.array((5000, self.array_size_x/2, self.array_size_y/2, self.array_size_x/5, self.array_size_y/8, 200), dtype=np.float32) 

        self.repeat = int(repeat)
        self.tolerance = tolerance
        self.max_num_iter = int(max_num_iter)

    def generate_data(self):

        data = self.gaussian_2d(*self.true_param)(*np.indices((self.array_size_x, self.array_size_y)))

        rng = np.random.default_rng(seed=0)
        data += (rng.random(data.shape)-0.5) * data / 2 # add nosie, element-wise product

        return data

    def gaussian_2d(self, amp, x_mean, y_mean, x_sigma, y_sigma, offset):

        return lambda x, y: amp*np.exp(-0.5*((x-x_mean)/x_sigma)**2-0.5*((y-y_mean)/y_sigma)**2) + offset
    
    def curve_fit_test(self, print_time=True):

        data = self.generate_data()

        elapsed_time_list = []
        for _ in range(self.repeat):
            t0 = time.time()

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

            # use optimize function to obtain 2D gaussian fit
            errorfunction = lambda p: np.ravel(self.gaussian_2d(*p)(X, Y)-data.astype(np.float32))
            popt, pcov, infodict, msg, success = scipy.optimize.leastsq(errorfunction, init_param, 
                                                   maxfev=self.max_num_iter, 
                                                   ftol=self.tolerance,
                                                   full_output=True)

            elapsed_time_list.append(time.time() - t0)
            assert success in [1,2,3,4], 'Scipy: Fit failed.\n ' + msg

        if print_time:
            print('Scipy elapsed times: ' + ', '.join(['{:.3f}'.format(i) for i in elapsed_time_list]) + ' s.')
            print('Scipy average time: {:.3f} s.'.format(np.mean(elapsed_time_list)))

        return (popt, self.true_param, init_param, data, self.gaussian_2d(*popt)(X, Y))
    
class gpufit_curve_fit_tester(scipy_curve_fit_tester):
    def __init__(self, array_size_x=None, array_size_y=None, repeat=None, tolerance=None, max_num_iter=None):

        kwargs = {}
        if array_size_x is not None:
            kwargs['array_size_x'] = array_size_x
        if array_size_y is not None:
            kwargs['array_size_y'] = array_size_y
        if repeat is not None:
            kwargs['repeat'] = repeat
        if tolerance is not None:
            kwargs['tolerance'] = tolerance
        if max_num_iter is not None:
            kwargs['max_num_iter'] = max_num_iter

        super().__init__(**kwargs)

        assert self.array_size_x == self.array_size_y, 'gpufit: array size x and y are different. Gpufit only supports square arrays.'

    def curve_fit_test(self, print_time=True):

        data = self.generate_data()

        exec_time_list = []
        elapsed_time_list = []
        for _ in range(self.repeat):
            t0 = time.time()

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
            
            init_param = np.array([[amp, x_mean, y_mean, x_sigma, y_sigma, offset]], dtype=np.float32)

            fitted_param, states, chi_squares, number_iterations, execution_time = gf.fit(data = data.T.reshape(1, -1).astype(np.float32), 
                                                                                        weights = None, 
                                                                                        model_id = gf.ModelID.GAUSS_2D_ELLIPTIC,
                                                                                        initial_parameters = init_param, 
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

        return (fitted_param[0], self.true_param, init_param[0], data, self.gaussian_2d(*fitted_param[0])(X, Y))

class torchimize_curve_fit_tester(scipy_curve_fit_tester):
    def __init__(self, array_size_x=None, array_size_y=None, repeat=None, tolerance=None, max_num_iter=None):

        kwargs = {}
        if array_size_x is not None:
            kwargs['array_size_x'] = array_size_x
        if array_size_y is not None:
            kwargs['array_size_y'] = array_size_y
        if repeat is not None:
            kwargs['repeat'] = repeat
        if tolerance is not None:
            kwargs['tolerance'] = tolerance
        if max_num_iter is not None:
            kwargs['max_num_iter'] = max_num_iter

        super().__init__(**kwargs)

        # self.init_param = torch.tensor(self.init_param, dtype=torch.float32, device='cuda')
        self.max_num_iter = torch.tensor(self.max_num_iter, dtype=torch.int32, device='cuda')
        self.tolerance = torch.tensor(self.tolerance, dtype=torch.float32, device='cuda')
    
    def gaussian_2d_torch(self, amp, x_mean, y_mean, x_sigma, y_sigma, offset):

        return lambda x, y: amp*torch.exp(-0.5*((x-x_mean)/x_sigma)**2-0.5*((y-y_mean)/y_sigma)**2) + offset
    
    def curve_fit_test(self, print_time=True):

        data_numpy = self.generate_data()
        data_torch = torch.tensor(data_numpy, dtype=torch.float32, device='cuda')

        cost_func = lambda p, x, y, data: data - self.gaussian_2d_torch(*p)(x, y)
        jac_func = lambda p, x, y, data: jacobian_approx_loop(p, f=cost_func, dp=1e-6, args=(x, y, data))

        elapsed_time_list = []
        for _ in range(self.repeat):
            t0 = time.time()
            
            total = data_numpy.sum()
            X, Y = np.indices(data_numpy.shape)
            X_torch = torch.tensor(X, dtype=torch.float32, device='cuda')
            Y_torch = torch.tensor(Y, dtype=torch.float32, device='cuda')
            x_mean = np.sum(X*data_numpy)/total
            x_mean = np.clip(x_mean, 0, data_numpy.shape[0]-1) # coerce x_mean to data shape
            y_mean = np.sum(Y*data_numpy)/total
            y_mean = np.clip(y_mean, 0, data_numpy.shape[1]-1) # coerce y_mean to data shape
            col = data_numpy[:, int(y_mean)]
            x_sigma = np.sqrt(np.abs((np.arange(col.size)-x_mean)**2*col).sum()/col.sum())
            row = data_numpy[int(x_mean), :]
            y_sigma = np.sqrt(np.abs((np.arange(row.size)-y_mean)**2*row).sum()/row.sum())
            offset = (data_numpy[0, :].sum()+data_numpy[-1, :].sum()+data_numpy[:, 0].sum()+data_numpy[:, -1].sum())/np.sum(data_numpy.shape)/2
            amp = data_numpy.max() - offset
            
            init_param = torch.tensor([amp, x_mean, y_mean, x_sigma, y_sigma, offset], dtype=torch.float32, device='cuda')

            # returns a list of tensors consisting of fitted parameters found in each iteration
            fitted_param = lsq_lma(init_param, function=cost_func, jac_function=jac_func, 
                                    args=(X_torch.ravel(), Y_torch.ravel(), data_torch.ravel()), 
                                    max_iter=self.max_num_iter, 
                                    # meth='marq', tau=1e-6, bet=10, gam=10,
                                    ftol=self.tolerance, gtol=self.tolerance, ptol=self.tolerance)
            
            torch.cuda.synchronize()
            elapsed_time_list.append(time.time() - t0)

        if print_time:
            print('PyTorch elapsed times: ' + ', '.join(['{:.3f}'.format(i) for i in elapsed_time_list]) + ' s.')
            print('PyTorch average time: {:.3f} s.'.format(np.mean(elapsed_time_list)))
    
        return (fitted_param[-1].cpu().numpy(), self.true_param, init_param.cpu().numpy(), 
                data_numpy, self.gaussian_2d(*fitted_param[-1].cpu().numpy())(X, Y))

class modified_torch_curve_fit_tester(torchimize_curve_fit_tester):
    def __init__(self, array_size_x=None, array_size_y=None, repeat=None, tolerance=None, max_num_iter=None):

        kwargs = {}
        if array_size_x is not None:
            kwargs['array_size_x'] = array_size_x
        if array_size_y is not None:
            kwargs['array_size_y'] = array_size_y
        if repeat is not None:
            kwargs['repeat'] = repeat
        if tolerance is not None:
            kwargs['tolerance'] = tolerance
        if max_num_iter is not None:
            kwargs['max_num_iter'] = max_num_iter

        super().__init__(**kwargs)
    
    def curve_fit_test(self, print_time=True):

        data_numpy = self.generate_data()
        data_torch = torch.tensor(data_numpy, dtype=torch.float32, device='cuda')

        elapsed_time_list = []
        for _ in range(self.repeat):
            t0 = time.time()
            
            total = data_numpy.sum()
            X, Y = np.indices(data_numpy.shape)
            X_torch = torch.tensor(X, dtype=torch.float32, device='cuda')
            Y_torch = torch.tensor(Y, dtype=torch.float32, device='cuda')
            x_mean = np.sum(X*data_numpy)/total
            x_mean = np.clip(x_mean, 0, data_numpy.shape[0]-1) # coerce x_mean to data shape
            y_mean = np.sum(Y*data_numpy)/total
            y_mean = np.clip(y_mean, 0, data_numpy.shape[1]-1) # coerce y_mean to data shape
            col = data_numpy[:, int(y_mean)]
            x_sigma = np.sqrt(np.abs((np.arange(col.size)-x_mean)**2*col).sum()/col.sum())
            row = data_numpy[int(x_mean), :]
            y_sigma = np.sqrt(np.abs((np.arange(row.size)-y_mean)**2*row).sum()/row.sum())
            offset = (data_numpy[0, :].sum()+data_numpy[-1, :].sum()+data_numpy[:, 0].sum()+data_numpy[:, -1].sum())/np.sum(data_numpy.shape)/2
            amp = data_numpy.max() - offset
            
            init_param = torch.tensor([amp, x_mean, y_mean, x_sigma, y_sigma, offset], dtype=torch.float32, device='cuda')

            # returns a list of tensors consisting of fitted parameters found in each iteration
            fitted_param, i, success = gaussian_2d_fit_lma(init_param, 
                                                X_torch, Y_torch, data_torch, 
                                                max_iter=self.max_num_iter, 
                                                # meth='marq', tau=1e-6, bet=10, gam=10,
                                                ftol=self.tolerance, gtol=self.tolerance, ptol=self.tolerance)
            
            torch.cuda.synchronize()
            elapsed_time_list.append(time.time() - t0)

        if print_time:
            print('Modified torch elapsed times: ' + ', '.join(['{:.3f}'.format(i) for i in elapsed_time_list]) + ' s.')
            print('Modified average time: {:.3f} s.'.format(np.mean(elapsed_time_list)))
    
        return (fitted_param.cpu().numpy(), self.true_param, init_param.cpu().numpy(), 
                data_numpy, self.gaussian_2d(*fitted_param.cpu().numpy())(X, Y))                     



# array_size = 100
scipy_test = scipy_curve_fit_tester()
fitted_param, true_param, init_param, data, scipy_fitted_data = scipy_test.curve_fit_test(print_time=True)

gpufit_test = gpufit_curve_fit_tester()
fitted_param, true_param, init_param, data, gpufit_fitted_data = gpufit_test.curve_fit_test(print_time=True)

torchimize_test = torchimize_curve_fit_tester(tolerance=1e-8, max_num_iter=100)
fitted_param, true_param, init_param, data, torchimize_fitted_data = torchimize_test.curve_fit_test(print_time=True)

modified_torch_test = modified_torch_curve_fit_tester(tolerance=1e-8, max_num_iter=100)
fitted_param, true_param, init_param, data, modified_torch_fitted_data = modified_torch_test.curve_fit_test(print_time=True)

h_list = []
legend_list = []
plt.imshow(data)

contour = plt.contour(scipy_fitted_data, colors='C0')
plt.clabel(contour, inline=1, fontsize=10)
h_scipy, _ = contour.legend_elements()
h_list.append(h_scipy[0])
legend_list.append('Scipy fit')

contour = plt.contour(gpufit_fitted_data, colors='C1')
plt.clabel(contour, inline=1, fontsize=10)
h_gpufit, _ = contour.legend_elements()
h_list.append(h_gpufit[0])
legend_list.append('Gpufit fit')

contour = plt.contour(torchimize_fitted_data, colors='C2')
plt.clabel(contour, inline=1, fontsize=10)
h_torchimize, _ = contour.legend_elements()
h_list.append(h_torchimize[0])
legend_list.append('torchimize fit')

contour = plt.contour(modified_torch_fitted_data, colors='C3')
plt.clabel(contour, inline=1, fontsize=10)
h_modified_torch, _ = contour.legend_elements()
h_list.append(h_modified_torch[0])
legend_list.append('modified torch fit')

plt.legend(h_list, legend_list, loc='upper right')
plt.show()