import roboticstoolbox as rt
import numpy as np
import scipy.io as scio
import kinpy as kp
import spatialmath as sm
import rtde_receive
import rtde_control
from tqdm import tqdm
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import time
import camera_class
from CSRL_orientation import *
from ActiVO_aux import *
import threading
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Declare math pi
pi = math.pi

# ip_robot = "192.168.1.60"     # for UR3e
ip_robot = "192.168.1.100"      # for UR5e

# Define robot
rtde_c = rtde_control.RTDEControlInterface(ip_robot)
rtde_r = rtde_receive.RTDEReceiveInterface(ip_robot)


# # commanded initial configuration for experimenting
# up UR3e
# q0d = np.array([-2.7242565790759485, -2.014890810052389, -1.7824010848999023, -0.8763185304454346, 1.5800056457519531, -1.227436367665426])

# side UR3e
# q0d = np.array([-3.6925345102893274, -2.9910251102843226, -1.1009764671325684, -1.0728790921023865, 2.3592722415924072, -2.264323059712545])

# side UR5e
# q0d = np.array([-0.41690332094301397, -1.883740564385885, -1.7535805702209473, -1.4085637044957657, 2.464108467102051, 1.0600205659866333])
# q0d = np.array([0.3244214355945587, -1.9122115574278773, -1.2255409955978394, -1.4460056287101288, 1.4586446285247803, 1.7281198501586914])
# q0d = np.array([1.632409930229187, 0.016547008151672316, 1.609415356312887, -3.1515056095519007, 0.759623110294342, 4.654907703399658])

# Ellipse exp
# q0d = np.array([-0.5215924421893519, -2.0054155788817347, -1.5068203210830688, -1.4493384447744866, 2.4010772705078125, 1.0605438947677612 - 1.57])
# q0d = np.array([-0.3801339308368128, -1.9773642025389613, -1.316433310508728, -1.7317115269102992, 2.124098539352417, -0.4773214499102991])

# Fruit exp
q0d = np.array([1.182332992553711, -2.2384849987425746, -1.71377694606781, -3.0127340755858363, 0.5584464073181152, 2.2159714698791504])
# q0d = np.array([0.7912634611129761, -1.980398794213766, -1.1590644121170044, -1.9489723644652308, 0.9563465714454651, 2.411074638366699 - 1.57])


# Move to the initial configuration
rtde_c.moveJ(q0d, 0.5, 0.5)

# Get initial configuration
q0 = np.array(rtde_r.getActualQ())

# Create the robot
# ur = rt.models.UR3()
# ur = rt.models.UR5()

# define the robot with its DH parameters
ur = rt.DHRobot([
    rt.RevoluteDH(d = 0.1625, alpha = pi/2),
    rt.RevoluteDH(a = -0.425),
    rt.RevoluteDH(a = -0.3922),
    rt.RevoluteDH(d = 0.1333, alpha = pi/2),
    rt.RevoluteDH(d = 0.0997, alpha = -pi/2),
    rt.RevoluteDH(d = 0.0996)
],name='UR5e')



# Get the initial robot state
R0c, p0c = get_robot_pose(ur, q0)

# print(R0c)
# print(p0c)
# asdsad[34] = 56


# Control cycle
dt = 0.002

# Init time
t = 0.0

# get time now
t_now = time.time()

# blocking
data_id = input('Press any key to start motion recording. This will be stored as the data ID: ')

# Uncomment if you want to free drive[-0.7225549856769007, -1.5372941692224522, -2.1203153133392334, -0.1600521367839356, 1.5096690654754639, 3.262672185897827]
# rtde_c.freedriveMode()

# THe distance from the camera
d  = 0.35

# COntrol gains
kp = 1.0
ko = 2.0
ka = 1.0

# This is the search window (in samples) for the synchronization
search_window = 500

# This is a flag for indicating the first iteration
firstTime = True

# Initialization of pcf_hat ... this passes through !!!!!!!! hstack !!!!!
pcf_hat = np.array([[0], [0], [0.3]])

# initialize filtered estimation
pcf_filtered = pcf_hat


# Initialize logging
pcflog = pcf_hat
tlog = np.array([t])

# Set the covariance matrix of the camera
sigma = 0.001
sigma_d = 0.05
Sigma_c = np.identity(3)
Sigma_c[0, 0] = sigma
Sigma_c[1, 1] = sigma
Sigma_c[2, 2] = sigma_d

# Total time is initialy set to a very large number (we don't know this before the first iteration)
T = 100000

# tr is the virtual time. Initialize it
tr = 0

# this is a position array for performing the dtw
pf_for_dtw = []

# this is the index to the reference trajectory
index_ref = 0

# this is the array of time of the reference signal
tr_array = np.zeros(10000)


# this is the initial pf reference for the DTW
p0c.shape = (3, 1)
pcf_hat.shape = (3, 1)
pf_ref = (p0c + R0c @ pcf_hat).T



# print(pf_ref)

# Initialization of Overline_Sigma_m
Sigma_m = np.identity(3)


index_array = np.array([0])
fig, axs = plt.subplots(3)
fig.suptitle('Synch')

# FOR EACH ITERATION (3 max are considered)
for i in range(2):

    # get the robot's state
    R0c, p0c = get_robot_pose(ur, q0)
    p0c.shape = (3, 1)

    print('Initiating iteration No: ', i)

    # Initialize the arrays that pass through !!!!!!!! hstack !!!!!
    Pcf_hat_iter = np.array([[0], [0], [0.3]])
    Qc_iter = np.array([[1], [0], [0], [0]])  # QUATERNION
    Pc_iter = p0c
    Pc_iter.shape = (3, 1)

    # if i == 0:


    if i == 1:
        # go to a new random configuration
        # for UR3e
        # q0d = np.array(
        #     [-2.7242565790759485, -2.014890810052389, -1.7824010848999023, -0.8763185304454346, 1.5800056457519531,
        #      -1.227436367665426])

        # for UR5e
        # exp ellipse
        # q0d = np.array([2.0775132179260254, -1.5693469533328717, 1.0836318174945276, -1.3998232048800965, -1.333468262349264, 5.02064323425293])
        # q0d = np.array(
        #     [0.7912634611129761, -1.980398794213766, -1.1590644121170044, -1.9489723644652308, 0.9563465714454651, 2.411074638366699 - 1.57])

        # exp fruit
        q0d = np.array([1.2330601215362549, -1.2422520977309723, -1.6271378993988037, -2.3346992931761683, 0.3470206558704376,
             0.5055640339851379])

        Sigma_m_now = Sigma_m[0:3, 0:3]
    # END OF IF


    # Move to the initial configuration
    rtde_c.moveJ(q0d, 0.5, 0.5)

    # Wait for a finger to appear
    while True:
        with tracker.lock:
            fingertips3d_result = tracker.get_fingertips3d()
        # END OF WITH

        time.sleep(0.002)
        # print(fingertips3d_result)
        if fingertips3d_result:
            break
        # END OF IF
    # END OF WHILE

    pcf_hat.shape = (3, 1)

    # after a finger is detected, wait for 3 seconds
    time.sleep(3)

    # make a boop sound for the humnan to start the demonstration
    beep = lambda x: os.system("echo -n '\a';sleep 0.015;" * x)
    beep(10)

    # initialize time and reference time
    t = 0
    tr = 0

    t_plot = 0.0

    # initialize reference index
    index_ref = 0
    index_ref_prev = -1

    # initialize finger position for DTW
    pf_for_dtw = (p0c + R0c @ pcf_hat).T

    # This variable is only for the anti-spike filter
    pcf_prev = pcf_hat

    # print(pf_for_dtw)
    tlog = np.array([t])

    # this is for accepting commands from keyboard
    with NonBlockingConsole() as nbc:

        # while the reference index is less that the size of the reference time array
        while index_ref < len(tr_array):     # CONTROL LOOP

            # Start control loop - synchronization with the UR
            t_start = rtde_c.initPeriod()

            # Integrate time
            t = t + dt
            tr = t

            # Get joint values
            q = np.array(rtde_r.getActualQ())

            # get state
            R0c, p0c = get_robot_pose(ur, q)

            # initialize v_1 and v_2
            v_1 = np.zeros(6)
            v_2 = np.zeros(6)



            # get the current estimation of the finger
            firstTime, pcf_hat, pcf_filtered = get_pcf(firstTime, pcf_filtered)

            # if firstTime:
            #     pcf_prev = pcf_hat
            # # END IF
            #
            # firstTime = firstTime2

            # Anti-spike filter
            # if math.sqrt((pcf_hat[0] - pcf_prev[0])**2+(pcf_hat[1] - pcf_prev[1])**2+(pcf_hat[2] - pcf_prev[2])**2) > 0.10:
            #     print('pcf_hat=', pcf_hat)
            #     print('pcf_prev=', pcf_prev)
            #     print('norm=', np.linalg.norm(pcf_hat - pcf_prev))
            #     pcf_hat = pcf_prev
            # # END OF IF
            pcf_prev = pcf_hat


            # compute v_2 for centering and maintaining distance
            norm_pcf_hat = np.linalg.norm(pcf_filtered)
            v_2[:3] = kp * (norm_pcf_hat - d) * R0c @ pcf_filtered / norm_pcf_hat
            v_2[-3:] = ko * R0c @ angle_axis_vectors(np.array([0, 0, 1]), pcf_filtered)

            # get the Jacobian
            J = get_jacobian(ur, q)

            # shape arrays before stacking them
            pcf_hat.shape = (3, 1)
            p0c.shape = (3, 1)
            R0c.shape = (3, 3)
            Qc = rot2quat(R0c)
            Qc.shape = (4, 1)

            # R0c, p0c = get_robot_UR_pose(rtde_r, ur)
            p0c.shape = (3, 1)

            # log time
            tlog = np.vstack((tlog, t))

            # compute Sigma now
            Sigma_now = R0c @ Sigma_c @ R0c.T

            # if we are in the first iteration
            if i == 0:

                # this is the reference finger position utilized for the DTW
                # pf_ref = np.vstack((pf_ref, p0c.T + R0c @ pcf_filtered))
                pf_ref = np.vstack((pf_ref, p0c.T + R0c @ pcf_filtered))

                # print("R0c @ pcf_filtered= ", (R0c @ pcf_filtered))

                print('p0c =', p0c)
                print('R0c =', R0c)
                print('pf_ur =', (p0c + R0c @ pcf_hat).T)
                print('pf_corke =', (p0c + R0c @ pcf_hat).T)
                print('pcf_hat =', pcf_hat)

                # here we stack an array with the initial overline Sigma now.
                # The array stackes the matrices horizontaly, i.e. it is of length 3 x 3*n (where n is the number of samples of first iteration)
                Sigma_m = np.hstack((Sigma_m, Sigma_now))

                # print("----------------------")
                # print('')


                # print("pcf_hat= ", pcf_hat)
                # print("Pcf_hat_iter= ", Pcf_hat_iter)

                # plot results

                # if t_plot > 0.5:
                #     for iplot in range(3):
                #         axs[iplot].plot(pf_ref[:, iplot], 'k', linewidth=2.0)
                #     plt.show()
                #     t_plot = 0
                # END IF plot

                t_plot = t_plot + dt

            else:

                # Compute current position of the finger
                # pf_now = p0c + R0c @ pcf_hat
                pf_now = p0c + R0c @ pcf_hat

                # create an array with the reference positions of the finger
                pf_ref_check = np.array([])

                # THe end index of this reference array is +search_window or until the end of the time series
                len_of_time_series = len(pf_ref[:, 0])
                end_Index = min(len_of_time_series, index_ref+search_window)

                # Uncomment to always search until the end
                # pf_ref_check = (pf_ref[index_ref:-1, :]).T

                # fill array
                pf_ref_check = (pf_ref[index_ref:end_Index, :]).T

                # In this compute the Euclidean norm
                norm_dif = np.zeros(len(pf_ref_check[0, :]))
                for jj in range(len(pf_ref_check[0, :])):
                    norm_dif[jj] = (pf_ref_check[0, jj] - pf_now[0])**2 + (pf_ref_check[1, jj] - pf_now[1])**2 + (pf_ref_check[2, jj] - pf_now[2])**2

                # Here we search for the next nearest of the current position to the reference using the norm
                index_ref = index_ref + np.argmin(norm_dif)

                print('index_ref: ', index_ref)
                print('intex_max" ', len(tr_array))

                index_array = np.vstack((index_array, index_ref))

                # Current overline_Sigma_m is taken from the corresponding slot
                if index_ref > index_ref_prev:
                    Sigma_m_now = np.array(Sigma_m[0:3, index_ref*3:index_ref*3+3])


                # Here we compute the eigenvalues and eigenvector
                evals, evec = np.linalg.eig(Sigma_m_now)
                # nd is the eigenvector with the largest eigenvalue (major axis)
                nd = evec[:, np.argmax(evals)]

                # If the z-axis of nd is negative, then take the opposite axis as the eigenvector,
                # this is done to ensure that the motions will remain above the supporting surface
                if nd[2] < 0:
                    nd = - nd

                # print('------------------')
                print('nd=', nd)

                # Option 1: Compute n from the projection
                # Tproj = R0c[0:3, 0:2]
                # n = Tproj @ (np.linalg.inv(Tproj.T @ Tproj)) @ Tproj.T @ nd
                # n = Tproj @ (np.linalg.inv(Tproj.T @ Tproj)) @ Tproj.T @ np.array([0, 0, 1])

                # Option 2: n is equal to the x axis of the camera (one of the minor axis of the ellipsoid)
                # n = - R0c[:, 0]
                n = R0c[:, 1]
                # If the z-axis of n is negative, then take the opposite,
                # this is done to ensure that the motions will remain above the supporting surface
                if n[2] < 0:
                    n = - n

                print('n=', n)
                #
                # asdsa[34] = 45

                # rtde_c.speedStop()



                # Compute the control signal v_1
                # pcf_filtered[0] = 0
                # pcf_filtered[1] = 0
                # pcf_filtered[2] = 0.3
                v_1[:3] = ka * skewSymmetric(R0c @ pcf_filtered) @ angle_axis_vectors(n, nd)
                v_1[-3:] = ka * angle_axis_vectors(n, nd)

                print('v1= ', v_1)

                # fill the array with data
                if index_ref > index_ref_prev:

                    # compute the new Sigma_m
                    Sigma_m_new = np.linalg.inv( np.linalg.inv(Sigma_m_now)  + np.linalg.inv(Sigma_now) )

                    # fill all elements until this instance with the new Sigma
                    j_fill = index_ref_prev + 1
                    while j_fill < index_ref + 1:
                        Sigma_m[0:3, j_fill * 3:j_fill * 3 + 3] = Sigma_m_new
                        j_fill = j_fill + 1
                    # Pcf_hat_iter[:, index_ref_prev+1:index_ref+1] = pcf_hat
                    # Qc_iter[:, index_ref_prev+1:index_ref+1] = Qc
                    # Pc_iter[:, index_ref_prev+1:index_ref+1] = p0c
                # END IF
                index_ref_prev = index_ref

                # plot results
                # fig, axs = plt.subplots(3)
                # fig.suptitle('Synch')
                # for iplot in range(3):
                #     axs[iplot].plot(pf_ref[:, iplot], 'k', linewidth=2.0)
                #     axs[iplot].plot(pf_now[:, iplot], 'r*', linewidth=2.0, markersize=4.0)
                #     axs[iplot].plot(pf_ref[0:index_ref, iplot], 'b', linewidth=3.0)
                # plt.show()

            # END OF IF-ELSE

            # stack data array
            Pcf_hat_iter = np.hstack((Pcf_hat_iter, np.array(pcf_hat)))
            # Qc_iter = np.hstack((Qc_iter, np.array(Qc)))


            Qc_iter = np.hstack((Qc_iter, Qc))
            # Pc_iter = np.hstack((Pc_iter, np.R0carray(p0c)))
            Pc_iter = np.hstack((Pc_iter, np.array(p0c)))
            # print("pcf_hat= ", pcf_hat)
            # print("Pcf_hat_iter= ", Pcf_hat_iter)





            # if the key a is pushed
            if nbc.get_data() == 'a' or index_ref > 0.97 * len(tr_array):  # x1b is ESC
                # stop the (n > 0) iteration

                if i>0:
                    # fill the final data
                    Pcf_hat_iter[:, index_ref_prev + 1:-1] = pcf_hat
                    # Qc_iter[:, index_ref_prev + 1:-1] = Qc
                    # Pc_iter[:, index_ref_prev + 1:-1] = p0c


                    Qc_iter[:, index_ref_prev + 1:-1] = Qc
                    Pc_iter[:, index_ref_prev + 1:-1] = p0c
                # END IF

                beep = lambda x: os.system("echo -n '\a';sleep 0.015;" * x)
                beep(5)


                print('Stopping the iteration')
                break
            # END OF IF

            # v_1 = np.zeros(6)
            # v_2 = np.zeros(6)

            # print('R0c: ', R0c)

            # Inverse kinematics mapping with siongularity avoidance
            qdot = np.linalg.pinv(J, 0.1) @ ( v_1 + v_2)

            ### Just for testing
            # qdot = np.zeros(6)

            # set joint speed with acceleration limits
            rtde_c.speedJ(qdot, 1.0, dt)

            # This is for synchronizing with the UR robot
            rtde_c.waitPeriod(t_start)

        # END OF WHILE -- control loop
    # END OF WITH





    # print('pf_ref: ',  pf_ref)
    # this is the end of the iteration
    # stop the robot
    rtde_c.speedStop()

    # if we are on the first iteration
    if i == 0:

        # TODO: Erase the first element?
        # Initialize the array including all measurements

        # print("Pcf_hat_iter = ", Pcf_hat_iter)

        # Pcf_hat_all = np.array(Pcf_hat_iter)
        # Qc_all = np.array(Qc_iter)
        # Pc_all = np.array(Pc_iter)

        # print("Pcf_hat_all = ", Pcf_hat_all)

        # print(len(Pcf_hat_all[0, :]))

        # the time reference array is the logged time in the first iteration
        tr_array = np.array(tlog)

        # ------------- Save "all" variable in a file
        data = {'Pcf_hat_iter': Pcf_hat_iter, 'Qc_iter': Qc_iter, 'Pc_iter': Pc_iter, 'tr': tr_array,
                'index_array': index_array, 'Sigma_m': Sigma_m}
        scio.savemat('Logging_' + str(data_id) + '_ref.mat', data)

        # -----------------PLOTS
        plt.plot(tr_array, pf_ref[:, 0])
        plt.plot(tr_array, pf_ref[:, 1])
        plt.plot(tr_array, pf_ref[:, 2])
        plt.show()

        # The total duration is defined by this first demo
        T = tr_array[-1]
    else:
        # print(len(Pcf_hat_all[0,:]))
        # print(len(Pcf_hat_iter[0,:]))

        # stack the arrays 3*m X Nsamples
        # Pcf_hat_all = np.vstack((Pcf_hat_all, Pcf_hat_iter))
        #
        # # stack the arrays 4*m X Nsamples
        # Qc_all = np.vstack((Qc_all, Qc_iter))
        #
        # # stack the arrays 3*m X Nsamples
        # Pc_all = np.vstack((Pc_all, Pc_iter))

        # ------------- Save "all" variable in a file
        data = {'Pcf_hat_iter': Pcf_hat_iter, 'Qc_iter': Qc_iter, 'Pc_iter': Pc_iter, 't': tlog,
                'index_array': index_array, 'Sigma_m': Sigma_m}
        scio.savemat('Logging_' + str(data_id) + '_' + str(i) + '.mat', data)


# END OF (BIG) FOR .... ITERATIONS

# stop the robot
rtde_c.speedStop()
rtde_c.stopScript()

# -----------------PLOTS
# ax = plt.axes(projection='3d')
# ax.plot3D(Pcf_hat_all[0][:][0], Pcf_hat_all[0][:][1], Pcf_hat_all[0][:][2])
# ax.set_aspect('equal', 'box')
# plt.show()








