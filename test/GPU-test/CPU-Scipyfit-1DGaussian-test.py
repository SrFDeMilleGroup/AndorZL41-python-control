import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import time

def generate_gauss_1d(x, a, x0, s, b):
    """
    Generates a 1D Gaussian curve.

    :param parameters: The parameters (a, x0, s, b)
    :param x: The x values
    :return: A 1D Gaussian curve.
    """

    rng = np.random.default_rng()
    y = a * np.exp(-np.square(x - x0) / (2 * s**2)) + b + (rng.random(x.shape)-0.5) * a / 1.5

    return y

def generate_gauss_1d_no_noise(x, a, x0, s, b):
    """
    Generates a 1D Gaussian curve.

    :param parameters: The parameters (a, x0, s, b)
    :param x: The x values
    :return: A 1D Gaussian curve.
    """

    y = a * np.exp(-np.square(x - x0) / (2 * s**2)) + b

    return y

n_points = int(1e6)
true_parameters = np.array((4, n_points/2, n_points/20, 1), dtype=np.float32)
x = np.arange(n_points, dtype=np.float32)
# x = np.linspace(0, 4, n_points, dtype=np.float32)
data = generate_gauss_1d(x, *true_parameters)
# tolerance
tolerance = 0.001

# max_n_iterations
max_n_iterations = 50

initial_parameters = [3, n_points/1.8, n_points/40, 0]

t0 = time.time()
for _ in range(1):
    popt, pcov, infodict, mesg, ier = curve_fit(generate_gauss_1d_no_noise, x, data, p0=initial_parameters, full_output=True, ftol=tolerance)
print(time.time() - t0)
print(infodict["nfev"])

plt.plot(x, data)
plt.plot(x, generate_gauss_1d_no_noise(x, *popt))
plt.show()