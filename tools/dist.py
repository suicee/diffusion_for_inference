from scipy.special import polygamma,digamma,gamma
from scipy.stats import gamma as gamma_dis
import numpy as np


class GammaModel:
    def __init__(self, N_obs=100):
        self.N_obs = N_obs

    def simulator(self, thetas):
        alpha, beta = thetas[:, :1], thetas[:, 1:]
        samples = np.random.gamma(alpha, beta, size=(thetas.shape[0], self.N_obs))
        return samples

    def log_likelihood(self, thetas, x):
        alpha, beta = thetas[:, :1], thetas[:, 1:]
        return gamma_dis.logpdf(x, alpha, scale=beta).sum(1)

    def fisher_score(self, thetas, x):
        alpha, beta = thetas[:, :1], thetas[:, 1:]
        grad_alpha = (-digamma(alpha) + np.log(x) - np.log(beta)).sum(1)
        grad_beta = (-alpha / beta + x / beta ** 2).sum(1)
        return np.stack([grad_alpha, grad_beta], 1)

    def fisher_information(self, thetas):
        alpha, beta = thetas[:, 0], thetas[:, 1]
        I_00 = polygamma(1, alpha)
        I_11 = alpha / (beta ** 2)
        I_01 = 1 / beta
        I_10 = 1 / beta
        return self.N_obs * np.stack([np.stack([I_00, I_01], 1), np.stack([I_10, I_11], 1)], 1)