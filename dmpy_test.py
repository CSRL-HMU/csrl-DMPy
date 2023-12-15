from kernel import *
from kernelBase import *
from CSRL_math import *
import scipy.io

from dmp import *

kernelType = 'Gaussian' # other option: sinc
canonicalType = 'linear' # other option: exponential



########### Test kernel
# k = kernel(kernelType, 1, 2)
# k.plot()

########### Test kernel base
# kb = kernelBase(10, 5, kernelType, canonicalType, 0.5)

# kb.plot_t()
# kb.plot_x()

########### Test dmp
data = scipy.io.loadmat('c:\python_WS\CSRL-DMPlib\example_data.mat')
pd = data['p1']

yd = pd[:,0]
y = np.zeros(yd.size)

dt = 0.001
t = np.array(list(range(yd.size))) * dt

dmpx = dmp(10, t[-1], kernelType, canonicalType)
dmpx.train(dt, yd)

state = np.array([0, yd[0], 0])
state_dot = np.zeros(3)

# dmpx.set_init_position(yd[0]+0.1)

# dmpx.set_goal(yd[-1]+0.1)

# dmpx.set_tau(0.5)

i = 0
for ti in t:
    state = state + state_dot * dt
    state_dot = dmpx.get_state_dot( state[0], state[1], state[2])
    y[i] = state[1]
    i = i + 1

plt.plot(t,yd,'r--')
plt.plot(t,y,'k-')

plt.xlabel('$t$(s)',fontsize=14 )
plt.ylabel('$y(t)$',fontsize=14 )
plt.title('DMP evolution')
plt.legend(('demonstrated', 'DMP'))
plt.xlim(0,t[-1])
plt.grid()
plt.show()
    

plt.show()
