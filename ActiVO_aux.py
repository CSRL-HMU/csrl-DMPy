import numpy as np
import sys, termios, tty
import select
import fcntl
import camera_class
import threading
from CSRL_orientation import *

# Initialize variables
filter_pole = 0.01
tracker = camera_class.HandTracker()
tracker.start_tracking()
# Get initial configuration

# This is a function for getting asynchronously a key from the keyboard
class NonBlockingConsole(object):

    keyFlag = False

    def __enter__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, type, value, traceback):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    def get_data(self):
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)

        return False

# this returns the finger position estimate from the camera
def get_pcf(firstTime, pcf_filtered):

    pcf_hat = np.array([0, 0, 0.3])

    # Get finger data
    with tracker.lock:
        fingertips3d_result = tracker.get_fingertips3d()

    # if the array is not empty
    if fingertips3d_result:

        # if this the first time, initialize the state of the filter
        if firstTime:
            pcf_hat = np.array(fingertips3d_result[0])
            pcf_filtered = pcf_hat

        # If the finger is within a cylinder of 0.1m around the center (z-axis)
        if np.array(fingertips3d_result[0])[0]**2 + np.array(fingertips3d_result[0])[1]**2 < 0.3**2:
            pcf_hat = np.array(fingertips3d_result[0])
            firstTime = False
        else:
            print("[get_pcf] Out of cylinder !!!!!!!!!!!!")

        # State equation of the filter with integration
        pcf_filtered = filter_pole * pcf_hat + (1 - filter_pole) * pcf_filtered

    return firstTime, pcf_hat, pcf_filtered

# this returns the robot's pose
def get_robot_pose(ur, q):

    # Get the hom. transform from robotics toolbox
    g = ur.fkine(q)
    # get rotation matrix
    R0e = np.array(g.R)
    # get translation
    p0e = np.array(g.t)

    print('R0e=', R0e)
    print('p0e=', p0e)

    # this is the pose of the camera with respec to the end-effector frame
    Rec = rotZ(pi/2)
    pec = np.array([0, -0.0325, 0.123])
    # pec = np.array([0.123, 0, 0])


    # compute the pose of the camera with respect to the inertial frame
    R0c = R0e @ Rec
    p0c = p0e + R0e @ pec

    return R0c, p0c

# this returns the robot's pose
def get_robot_UR_pose(rtde_r, ur):

    print("UR pose= ", rtde_r.getActualTCPPose())
    p = (rtde_r.getActualTCPPose())[0:3]

    # rx = 2.49
    # ry = -2.81
    # rz = -0.88
    # R = rxryrz_to_rotation((rtde_r.getActualTCPPose())[3], (rtde_r.getActualTCPPose())[4], (rtde_r.getActualTCPPose())[5])
    # R = angle2rot([(rtde_r.getActualTCPPose())[3], (rtde_r.getActualTCPPose())[4], (rtde_r.getActualTCPPose())[5]])

    # Get the hom. transform from robotics toolbox
    g = ur.fkine(rtde_r.getActualQ())

    # get rotation matrix
    R0e = np.array(g.R)
    # this is the pose of the camera with respec to the end-effector frame
    Rec = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    # pec = np.array([0.123, 0, 0])
    # compute the pose of the camera with respect to the inertial frame
    R0c = R0e @ Rec

    print("UR p= ", rotZ(pi) @ p)
    print("UR R= ", R0c)

    return R0c, rotZ(pi) @ p


# returns the Jacobian of the robot (manera frame {C}) w.r.t. the world frame
def get_jacobian(ur, q):

    # get the Jacobian from robotics toolbox
    J = np.array(ur.jacob0(q))

    # get pose of the robot
    g = ur.fkine(q)
    p0e = np.array(g.t)
    R0e = np.array(g.R)

    pec = np.array([0, -0.0325, 0.123])
    p0c = p0e + R0e @ pec

    pce = p0e - p0c

    # COmpute the Jacobian for the cmaera frame
    GammaCE = np.identity(6)
    GammaCE[:3, -3:] = skewSymmetric(pce)
    J = GammaCE @ J

    return J
