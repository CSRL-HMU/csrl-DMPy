from dmp import *
from orientation import *
from CSRL_math import *
import numpy as np

import scienceplots
import matplotlib.pyplot as plt

# latex style plots 
plt.style.use(["default","no-latex"])

class dmpSE3:
    def __init__(self, N_in, T_in, kernelType_in, xType_in, a_x_in = 1.0, a_in=1.0, b_in=4.0, tau_in=1.0):
        self.kernelType = kernelType_in
        self.xType = xType_in
        self.a_x = a_x_in
        self.a = a_in
        self.b = b_in
        self.tau = tau_in

        self.dmp_translation = []
        self.dmp_orientation = []
        for i in range(3):
            temp_dmp = dmp(N_in, T_in, kernelType_in, xType_in, a_x_in, a_in, b_in, tau_in)
            self.dmp_translation.append(temp_dmp)
            self.dmp_orientation.append(temp_dmp)

    # inputs p_array: 3xN (translation), Q_array: 4xN (orientation in quaternion) 
    def train(self, dt, p_array, Q_array, plotPerformance = False):

       
        # number of samples
        N = p_array[0,:].size

        # translation target 
        self.p_target = p_array[:,-1]

        for i in range(3):
            self.dmp_translation.train(dt, p_array[i,:], False)

        # orientation target 
        self.Q_target = Q_array[:,-1]

        # compute the logarithmic error for each time instance 
        e_log_array = np.array((3,N))
        for i in range(Q_array[0,:].size):
            e_log_array[:,i] = logError(self.Q_target, Q_array[:,i])

        # computethe custom scaling factor in each axis 
        self.targetError = logError(Q_array[:,-1],Q_array[:,0])

        # train orientation DMP
        for i in range(3):
            self.dmp_orientation.train(dt, e_log_array[i,:], False, True, self.targetError[i])

        if plotPerformance: self.plotPerformance(dt, p_array, e_log_array)

    def get_state_dot(self, x, p, pdot, eOr, eOr_dot):
        
        p_out_dot = np.zeros(3)
        p_out_2dot = np.zeros(3)

        eOr_out_dot = np.zeros(3)
        eOr_out_2dot = np.zeros(3)

        for i in range(3):
            x_out_dot, p_out_dot[i], p_out_2dot[i] = self.dmp_translation[i].get_state_dot( x, p[i], pdot[i]) 
            x_out_dot, eOr_out_dot[i], eOr_out_2dot[i] = self.dmp_orientation[i].get_state_dot( x, eOr[i], eOr_dot[i], True, self.targetError[i]) 

        return x_out_dot, p_out_dot, p_out_2dot, eOr_out_dot, eOr_out_2dot
      

    # sets the initial state p and A (can be quaternion or rotation matrix)
    def set_init_position(self, p0_in, A0_in):

        self.p0 = p0_in
        self.Q0 = enforceQuat(A0_in)

        # calculate initial error from the target 
        self.targetError = logError(self.Q_target, self.Q0)



    # sets the goal fro translation and orientation (can be quaternion or rotation matrix)
    def set_goal(self, gP_in, gOr_in):

        self.p_target = gP_in
        self.Q_target = enforceQuat(gOr_in)

        # recalculate the initial orientation error 
        self.targetError = logError(self.Q_target, self.Q0)

        
    # sets time scaling parameter for all axes
    def set_tau(self, tau_in):
        self.tau = tau_in


    def plotPerformance(self, dt, p_array, e_log_array):
        
       
        t = np.array(list(range(p_array[0,:].size))) * dt

        if self.xType == 'linear':
            x = 0
        else:
            x = 1

        x_dot = 0

        p0 = p_array[:,0]
        elog0 = e_log_array[:,0]

        state = [p0, elog0, 0, 0]
        state_dot = np.zeros(4)

        pDMP = np.zeros((3,t.size))
        eOrDMP = np.zeros((3,t.size))

        i = 0
        for ti in t:
            x = x + x_dot * dt
            state = state + state_dot * dt
            x_dot, state_dot[0], state_dot[1], state_dot[2], state_dot[3] = self.get_state_dot( x, state[0], state[1], state[2], state[3])
            pDMP[:,i] = state[0]
            eOrDMP[:,i] = state[2]
            i = i + 1


        # plot results
        fig = plt.figure(figsize=(4, 6))


        for i in range(3):
            axs = fig.add_axes([0.21, ((5-i)/6)*0.9+0.1, 0.7, 0.12])
            axs.plot(t, pDMP[i, :], 'k', linewidth=1.0)
            axs.plot(t, p_array[i, :], 'k--', linewidth=2.0)
            axs.set_xlim([0, t[-1]])
            axs.set_ylabel('$p_' + str(i+1) + '(t)$ [m]',fontsize=14 )
            axs.set_xticks([])
        
        for i in range(3):
            axs = fig.add_axes([0.21, ((5-(i+3))/6)*0.9+0.1, 0.7, 0.12])
            axs.plot(t, eOrDMP[i, :], 'k', linewidth=1.0)
            axs.plot(t, e_log_array[i, :], 'k--', linewidth=2.0)
            axs.set_xlim([0, t[-1]])
            axs.set_ylabel('$e_{log,' + str(i+1) + '}(t)$ [rad]',fontsize=14 )
            
            if i==3:
                axs.set_xlabel('Time (s)',fontsize=14 )
                lgnd = axs.legend(['$DMP$','$demo$'],fontsize=11,ncol=2,loc="lower right")
                lgnd.get_frame().set_alpha(None)
            else:
                axs.set_xticks([])


        plt.show()

        


        

  
