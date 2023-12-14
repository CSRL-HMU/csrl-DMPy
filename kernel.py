import math
from CSRL_math import *


class kernel:

    type = 'Gaussian'
    h = 1.0 # for sinc type this is 1/Dt 
    c = 0.0
    

    def __init__(self, type_in, h_in, c_in):
        self.type = type_in
        self.h = h_in
        self.c = c_in


    def psi(self, x):
        if self.type == 'sinc':
            return sinc( self.h * pi * (x - self.c) )
        else:
            return math.exp(- self.h * ( x - self.c ) * ( x - self.c ) )
        

