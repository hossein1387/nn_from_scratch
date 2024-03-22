from typing import Any
import numpy as np

class BatchNorm:
    def __init__(self, running_mean, running_var, gamma, beta, eps=1e-5):
        self.running_mean = running_mean
        self.running_var = running_var
        self.eps = eps
        self.gamma = gamma
        self.beta = beta
    def forward(self, x):
        mu = np.mean(x)
        var = np.mean(np.pow((x-mu), 2))
        x = ((x-mu)/(np.sqrt(var+self.eps)))*self.gamma + self.beta
        return x
