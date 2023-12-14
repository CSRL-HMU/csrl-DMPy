from kernelBase import *
from CSRL_math import *
import numpy as np
import scienceplots
import matplotlib.pyplot as plt

# latex style plots 
plt.style.use(["default","no-latex"])

class dmp:

    def __init__(self, N_in, T_in, kernelType_in, xType_in, a_x_in = 1.0, a_in=1.0, b_in=4.0, tau_in=1.0):
        self.N = N_in
        self.T = T_in
        self.kernelType = kernelType_in
        self.xType = xType_in
        self.a_x = a_x_in
        self.a = a_in
        self.b = b_in
        self.Psi = kernelBase(N_in, T_in, kernelType_in, xType_in, a_x_in)
        self.W = np.zeros(N_in)

        # initial state of the canonical system
        if xType_in == 'linear':
            self.x0 = 0.0
        else: # canonical system is exponential
            self.x0 = 1.0
            
        self.tau = tau_in

