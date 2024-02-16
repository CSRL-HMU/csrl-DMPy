import numpy as np
from CSRL_orientation import *
import pathlib
import scipy.io
import matplotlib.pyplot as plt
import os
from dmpR3 import *



dt = 0.002

# load data
exp_id = 'ellipse'
# exp_id = 'fruitPicking'

folderPath = pathlib.Path(__file__).parent.resolve()

if os.name == 'nt': # the OS is Windows
    data = scipy.io.loadmat(str(folderPath) +'\\Logging_' + exp_id + '_ref.mat')
else:   # the OS is Linux
    data = scipy.io.loadmat(str(folderPath) +'/Logging_' + exp_id + '_ref.mat')

pc_0 = data['Pc_iter'][:,1:-1]
Qc_0 = data['Qc_iter'][:,1:-1]
pcf_hat_0 = data['Pcf_hat_iter'][:,1:-1] 
Sigma_0 = data['Sigma_m'][:,3:-1]
t_0 = (np.array(data['tr']).T)[0][0:-2]


N_0 = len(t_0)
pf_0 = np.zeros((3, N_0))

for i in range(N_0):
   Rc = quat2rot(Qc_0[:, i])
   pf_0[:, i] = pc_0[:, i] + Rc @ pcf_hat_0[:, i]

if os.name == 'nt': # the OS is Windows
    data = scipy.io.loadmat(str(folderPath) +'\\Logging_' + exp_id + '_1.mat')
else:   # the OS is Linux
    data = scipy.io.loadmat(str(folderPath) +'/Logging_' + exp_id + '_1.mat')

pc_1 = data['Pc_iter'][:,1:-1]
Qc_1 = data['Qc_iter'][:,1:-1]
pcf_hat_1 = data['Pcf_hat_iter'][:,1:-1]
Sigma_1 = data['Sigma_m'][:,3:-1]
t_1 = (np.array(data['t']).T)[0][0:-2]
index_1 = (np.array(data['index_array']).T)[0][1:-1]

N_1 = len(t_1)
pf_1 = np.zeros((3, N_1))


for i in range(N_1):
   Rc = quat2rot(Qc_1[:, i])
   pf_1[:, i] = pc_1[:, i] + Rc @ pcf_hat_1[:, i]

pf_1_aligned = np.zeros((3, N_0))
pc_1_aligned = np.zeros((3, N_0))
Qc_1_aligned = np.zeros((4, N_0))
pcf_hat_1_aligned = np.zeros((3, N_0))

j = 0
t_1_now = t_1[j]/t_1[-1]


##### simple time scaling
# for i in range(N_0):
#     t_0_now = t_0[i]/t_0[-1]
#     while t_1_now < t_0_now:
#         t_1_now = t_1[j]/t_1[-1]
#         j = j + 1
#     j = min(j , N_1-1)
#     pf_1_aligned[:, i] = pf_1[:, j]
#     pc_1_aligned[:, i] = pc_1[:, j]
#     Qc_1_aligned[:, i] = Qc_1[:, j]
#     pcf_hat_1_aligned[:, i] = pcf_hat_1[:, j]


# Exploiting on-line index finding
ind = 0
for j in range(N_1):
    for i in range(ind, index_1[j]+1):
        pf_1_aligned[:, i] = pf_1[:, j]
        pc_1_aligned[:, i] = pc_1[:, j]
        Qc_1_aligned[:, i] = Qc_1[:, j]
        pcf_hat_1_aligned[:, i] = pcf_hat_1[:, j]
    ind = index_1[j]

for i in range(ind, N_0):
    pf_1_aligned[:, i] = pf_1[:, -1]
    pc_1_aligned[:, i] = pc_1[:, -1]
    Qc_1_aligned[:, i] = Qc_1[:, -1]
    pcf_hat_1_aligned[:, i] = pcf_hat_1[:, -1]


t = t_0/t_0[-1]
N = len(pcf_hat_0[1, :])


Sigma_c = np.diag((0.02, 0.02, 0.5))

W = np.zeros((6, 6))
W[0:3, 0:3] = np.linalg.inv(Sigma_c)
W[3:6, 3:6] = np.linalg.inv(Sigma_c)
# W = np.identity(6)

A = np.zeros((6, 3))
B = np.zeros((6, 6))

Pc = np.zeros(6)
Pcf = np.zeros(6)

pf_hat = np.zeros((3,N))

for i in range(N):

    Rc_0 = quat2rot(Qc_0[:,i])
    Rc_1 = quat2rot(Qc_1_aligned[:,i])

    A[0:3, 0:3] = Rc_0.T
    A[3:6, 0:3] = Rc_1.T

    B[0:3, 0:3] = Rc_0.T
    B[3:6, 3:6] = Rc_1.T

    Pc[0:3] = pc_0[:,i]
    Pc[3:6] = pc_1_aligned[:,i]

    Pcf[0:3] = pcf_hat_0[:,i] 
    Pcf[3:6] = pcf_hat_1_aligned[:,i]

    #pf_hat[0:3, i] = (pf_0[:, i] + pf_1_aligned[:, i])/2
  
    pf_hat[0:3, i] = np.linalg.inv(A.T @ W @ A ) @ A.T @ W @ ( B @ Pc + Pcf)



########### DMP
    
p_train = pf_hat

dt = 0.002
t = np.array(list(range(p_train[1,:].size))) * dt

dmpPos = dmpR3(20, t[-1])
dmpPos.train(dt, p_train, False)


# p_0_offset = np.array([0.1, -0.2, 0.05])
p_goal_offset = np.array([0.00, 0.0, 0.00])


p_0_offset = np.zeros(3)
# p_goal_offset = np.zeros(3)

p0 = p_train[:,0] + p_0_offset
pT = p_train[:,-1] + p_goal_offset

dmpPos.set_init_state(p0)
dmpPos.set_goal(pT)

dmpPos.set_tau(1)

# dmpPos.plotResponse(dt,p0,2500)

p = p0
x = 0

p_dmp = np.zeros((3,N))

dp = np.zeros(3)
ddp = np.zeros(3)
dx = 0

# print(N)
# sim
for i in range(N):

    p = p + dp * dt
    dp = dp + ddp * dt
    x = x + dx * dt

    
    dx, dp, ddp = dmpPos.get_state_dot(x, p, dp)
    p_dmp[:, i] = p.T



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(pf_0[0, :], pf_0[1, :], pf_0[2, :], label='$\mathbf{p}_{f,0}$')
ax.plot(pf_1[0, :], pf_1[1, :], pf_1[2, :], label='$\mathbf{p}_{f,1}$')
ax.plot(pf_hat[0, :], pf_hat[1, :], pf_hat[2, :], label='$\hat{\mathbf{p}}_f$ (ActIVO)')
ax.plot(p_dmp[0, :], p_dmp[1, :], p_dmp[2, :], label='$\mathbf{p}_{dmp}$ (ActIVO)')

plt.axis('equal')
plt.legend()
plt.title('Final Signals')
plt.show()
                      