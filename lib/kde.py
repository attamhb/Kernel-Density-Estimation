import scipy
from scipy.stats import norm
import numpy as np
from scipy import linalg
from scipy.interpolate import interp1d

np.random.seed(42)
# fit Gaussian mixture
# https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html

############################################################################

##############################################################################
class KdeRegularizer:
    def __init__(self, X_sample, x_grid, kernel_type="normal"):
        self.X_sample = X_sample
        self.x_grid = x_grid
        self.N = len(X_sample)
        self.kernel_type = kernel_type

    def __call__(self, h):  # h bandwith
        # return f_hat(x;h) = 1/N*∑_{i=1}^N K_h(x - X_i)
        #   with x = x_grid, K_h(x) = 1/h*K(x/h)
        # std_h = np.sqrt(h)
        s = self.kde_f(self.x_grid, self.X_sample[0], h)
        for k in range(1, self.N):
            s += self.kde_f(self.x_grid, self.X_sample[k], h)
        return s / self.N

    def kde_f(self, x, x_i, h):
        return 1/h*self.kernel_f((x - x_i) / h, self.kernel_type)

    # from table 1 in chapte 1 of main.pdf
    def kernel_f(self, x, kernel_type):
        if kernel_type == "normal":  # default gaussain kernel, mu=0, sig=1
            k = norm.pdf(x, loc=0, scale=1)
        elif kernel_type == "uniform":  # for uniform kernel
            k = 0.5 * (np.abs(x) <= 1)
        elif kernel_type == "triangle":  # for traingular kernel
            k = (1 - abs(x)) * (np.abs(x) <= 1)
        elif kernel_type == "epanech":  # for Epanech kernel
            k = (3 / 4) * (1 - x ** 2) * (np.abs(x) <= 1)
        elif kernel_type == "quartic":  # for quartic kernel
            k = (15 / 16) * ((1 - x ** 2) ** 2) * (np.abs(x) <= 1)
        elif kernel_type == "triweight":  # for triweight kernel
            k = (35 / 32) * ((1 - x ** 2) ** 3) * (np.abs(x) <= 1)
        return k
