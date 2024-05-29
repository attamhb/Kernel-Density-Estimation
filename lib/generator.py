import numpy as np

np.random.seed(42)
# fit Gaussian mixture
# https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html

############################################################################
class GenerateSample:
    def __init__(self, weight, mu, var):
        self.weight = weight / sum(weight)
        self.mu = mu
        self.std = np.sqrt(var)
        self.var = var

    def sample(self, N):
        idx = np.random.choice(len(self.weight), N, p=self.weight)
        X = np.array(
            [(self.mu[idx_k] + self.std[idx_k] * np.random.randn()) for idx_k in idx]
        )
        return X

    def pdf(self, x_grid):
        s = self.weight[0] * self.normal(x_grid, self.mu[0], self.var[0])
        for k in range(1, len(self.weight)):
            s += self.weight[k] * self.normal(x_grid, self.mu[k], self.var[k])
        return s

    def normal(self, x, mu_scalar, var_scalar):  # auxiliary function to determine pdf
        return np.exp(-((x - mu_scalar) ** 2) / (2 * var_scalar)) / np.sqrt(
            2 * np.pi * var_scalar
        )
