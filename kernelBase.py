from CSRL_math import *
from kernel import *
import numpy as np

class kernelBase:

    def __init__(self, N_in, T_in, kernelType_in, xType_in, a_x_in = 1.0):
        
        # Number of kernels
        self.N = N_in

        # Canonical system type. Options:
        #       'linear' 
        #       'exponential'  (default)
        self.xType = xType_in

        # Pole of the exponential canonical system
        self.a_x = a_x_in

        # Total duration of motion 
        self.totalTime = T_in

        #  Kernel types. Options: 
        #      'Gaussian' (default)
        #      'sinc'
        self.kernelType = kernelType_in

       
        # The kenrel centers are set in time
        self.c_t = np.linspace(0, self.totalTime, self.N)

        # The kenrel widths are set in times
        if self.kernelType == 'sinc':
            self.h_t = 1 / (self.c_t[2] - self.c_t[1]) 
        else: # the kernel type is 'Gaussian'
            self.h_t = 1 / math.pow( self.c_t[2] - self.c_t[1] , 2) 

        self.kernelArray = []
        # Create the kernel array
        for i in range(self.N):  
            k = kernel(self.kernelType, self.h_t , self.c_t[i])
            self.kernelArray.append(k)


    # mapping time -> phase variable
    def ksi(self, t_in):
        if self.xType == 'linear':
            if t_in < self.totalTime:
                return t_in / self.totalTime
            else: 
                return 1
        else: # # canonical system is exponential 
            return math.exp(-  self.a_x  * t_in )
        
    # inverse mapping phase variable -> time
    def ksi_inv(self, x_in):
        if self.xType == 'linear':
            return x_in * self.totalTime
        else: # # canonical system is exponential 
            return - math.log( x_in ) / self.a_x
        
    # Reurns the base values in an array for specific phase variable value (x)
    def get_psi_vals(self, x_in):

        # phase var -> time
        t = self.ksi_inv(x_in)

      
        # init psi_vals
        psi_vals = np.zeros(self.N)
        for i in range(self.N):
            # get kernel value for this instance
            psi_vals[i] = self.kernelArray[i].psi(t)

        # return array
        return psi_vals
    
   



    
