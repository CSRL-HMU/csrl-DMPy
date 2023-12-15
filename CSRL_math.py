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

# returns the 5th order polynomial. It can take vectors
def get5thOrder(t, p0, pT, totalTime):
    p0 = np.array(p0)
    pT = np.array(pT)

    retTemp = np.zeros((p0.size, 3))

    if t<0:
    
        # before start
        retTemp[:,0] = p0
    
    elif t > totalTime:
    
        # after the end
        retTemp[:,0] = pT
    
    else:
        # somewhere betweeen ...
        # position
        retTemp[:,0] = p0 + (pT - p0) * (10 * pow(t / totalTime, 3) - 15 * pow(t / totalTime, 4) + 6 * pow(t / totalTime, 5))
       
        # vecolity
        retTemp[:,1]  = (pT - p0) * (30 * pow(t, 2) / pow(totalTime, 3) - 60 * pow(t, 3) / pow(totalTime, 4) + 30 * pow(t, 4) / pow(totalTime, 5))
        # acceleration
        retTemp[:,2]  = (pT - p0) * (60 * t / pow(totalTime, 3) - 180 * pow(t, 2) / pow(totalTime, 4) + 120 * pow(t, 3) / pow(totalTime, 5))
  
    return retTemp

# returns the 5th order polynomial. It can take vectors
def sigmoid(x, c, h):
    
  
    y = get5thOrder(x-(c-0.5*h), 0, 1, h)

    return y[0,0]
    
        


