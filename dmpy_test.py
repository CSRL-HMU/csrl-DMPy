from kernel import *
from kernelBase import *

# k = kernel('Gaussian', 1, 2)
# k = kernel('sinc', 1, 2)
# k.plot()

# kb = kernelBase(10, 5, 'sinc', 'linear')
kb = kernelBase(10, 5, 'Gaussian', 'linear')
# kb = kernelBase(10, 5, 'sinc', 'exponential', 0.5 )
# kb = kernelBase(10, 5, 'Gaussian', 'exponential', 0.5)
kb.plot_t()
# kb.plot_x()
