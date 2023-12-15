import math
import numpy as np

# 3.14159...
pi = math.pi

# the sinc function
def sinc(x):
    if x == 0: 
        return 1.0
    else:
        return math.sin(x)/x 
        

# the moving average filter (non-causal)
def maFilter(x, ncoeffs):
   
    MA_coeffs = np.ones(ncoeffs)/ncoeffs
    y_f = np.convolve(x, MA_coeffs)
    y = y_f[-x.size:]

    return y
        


