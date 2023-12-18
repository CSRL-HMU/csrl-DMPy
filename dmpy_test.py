from kernel import *
from kernelBase import *
from CSRL_math import *
from orientation import *
import scipy.io
import pathlib
import os


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

########### Test 5th order
# t = np.array(list(range(5000))) * 0.002
# p_dp_ddp = np.zeros((3,t.size))
# p0 = 1
# pT = 5

# i = 0
# for ti in t:
#     p_dp_ddp[:,i] = get5thOrder(ti, p0, pT, t[-1])
#     i = i + 1


# plt.plot(t,p_dp_ddp[0,:],'k')
# plt.plot(t,p_dp_ddp[1,:],'r')
# plt.plot(t,p_dp_ddp[2,:],'b')


# plt.xlabel('$t$(s)',fontsize=14 )
# plt.ylabel('$p(t)$',fontsize=14 )
# plt.title('5th order polynomial')
# plt.legend(('$p$', '$\dot{p}$', '$\ddot{p}$'))
# plt.xlim(0,t[-1])
# plt.grid()
# plt.show()


########### Test sigmoid
# t = np.array(list(range(5000))) * 0.002
# y = np.zeros(t.size)

# i = 0
# for ti in t:
#     y[i] = sigmoid(ti, 4, 0.5)
#     i = i + 1


# plt.plot(t,y,'k')

# plt.xlabel('$t$(s)',fontsize=14 )
# plt.ylabel('$y(t)$',fontsize=14 )
# plt.title('Sigmoid')
# plt.xlim(0,t[-1])
# plt.grid()
# plt.show()
    

########### Test orientation transitions 
# R = np.array([[ 0, 1, 0],
#               [-1, 0, 0],
#               [ 0, 0, 1]])

# print('R=',R)
# print('Q=',rot2quat(R))
# print('R=',quat2rot(rot2quat(R)))


########### Test quaternion functions 
# R = np.identity(3)
# Rd = rotZ(pi-0.001) 

# # Q = rot2quat(R)
# # Qd = rot2quat(Rd)

# print('[log] e=',logError(R,Rd))
# print('[vect] e=',vectorError(R,Rd))





########### Test single axis dmp
folderPath = pathlib.Path(__file__).parent.resolve()

if os.name == 'nt': # the OS is Windows
    data = scipy.io.loadmat(str(folderPath) +'\\example_data.mat')
else:   # the OS is Linux
    data = scipy.io.loadmat(str(folderPath) +'/example_data.mat')

pd = data['p1']

yd = pd[:,1]
y = np.zeros(yd.size)

dt = 0.001
t = np.array(list(range(yd.size))) * dt

dmpx = dmp(40, t[-1], kernelType, canonicalType)
dmpx.train(dt, yd)

# y_offset = 0.1
# goal_offset = 0.1
# tau = 0.7
y_offset = 0.0
goal_offset = 0.0
tau = 1


state = np.array([0, yd[0]+y_offset, 0])
state_dot = np.zeros(3)

dmpx.set_init_position(yd[0]+y_offset)
dmpx.set_goal(yd[-1]+goal_offset)
dmpx.set_tau(tau)

i = 0
for ti in t:
    state = state + state_dot * dt
    state_dot[0], state_dot[1], state_dot[2] = dmpx.get_state_dot( state[0], state[1], state[2])
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
    

