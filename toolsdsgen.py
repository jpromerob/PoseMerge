
import socket
import sys
import signal
import random
from ctypes import *
import numpy as np
import multiprocessing 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams['toolbar'] = 'None' 
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import multivariate_normal
import matplotlib.animation as animation
from scipy import stats
import time
import math
import datetime
import os
import random
import cv2
import pickle
import struct ## new
import zlib

def set_focal_lengths():

    focl = np.zeros((2,3))

    focl[0,0] = 649.229848
    focl[0,1] = 712.990500
    focl[0,2] = 780.709664
    focl[1,0] = 647.408499
    focl[1,1] = 712.531562
    focl[1,2] = 778.849697

    return focl

# Poses of virtual cameras with respect to their corresponding real camera space
def set_vir_poses(angles):
    v_poses = np.zeros((3,6))
    for k in range(3): # for cameras 1|2|3
        v_poses[k,0] = 0
        v_poses[k,1] = 0
        v_poses[k,2] = 0
        v_poses[k,3] = (math.pi/180)*(angles[1,k])    # around X axis -->  ang(YZ)
        v_poses[k,4] = (math.pi/180)*(-angles[0,k])   # around Y axis --> -ang(XZ)
        v_poses[k,5] = 0 

    return v_poses


'''
This function sets the camera poses based on manual readings from optitrack (using camera marker 'hat')
'''
def set_cam_poses():

    cam_poses = np.zeros((3,6))

    # Cam 1
    cam_poses[0,0] = -0.099 # cam1:cx
    cam_poses[0,1] = 0.968 # cam1:cy
    cam_poses[0,2] = 1.363 # cam1:cz
    cam_poses[0,3] = (math.pi/180)*(-71.499) # cam1:alpha
    cam_poses[0,4] = (math.pi/180)*(16.753) # cam1:beta
    cam_poses[0,5] = (math.pi/180)*(-20.992) # cam1:gamma

    # Cam 2
    cam_poses[1,0] = -0.570 # cam2:cx
    cam_poses[1,1] = 0.970 # cam2:cy
    cam_poses[1,2] = 1.395 # cam2:cz
    cam_poses[1,3] = (math.pi/180)*(-62.113) # cam2:alpha
    cam_poses[1,4] = (math.pi/180)*(-42.374) # cam2:beta
    cam_poses[1,5] = (math.pi/180)*(-6.134) # cam2:gamma

    # Cam 3
    cam_poses[2,0] = -0.664 # cam3:cx
    cam_poses[2,1] =  0.979 # cam3:cy
    cam_poses[2,2] =  0.538 # cam3:cz
    cam_poses[2,3] = (math.pi/180)*(148.698)# cam3:alpha
    cam_poses[2,4] = (math.pi/180)*(-46.056)# cam3:beta
    cam_poses[2,5] = (math.pi/180)*(148.752)# cam3:gamma

    return cam_poses

    
''' Translation Matrices'''
def get_transmats(cam_poses):
    
    mat_t_from = np.zeros((4,4,3))
    mat_t_to = np.zeros((4,4,3))
    for i in range(3): # Cam 1, 2, 3
        
        cx = cam_poses[i,0]
        cy = cam_poses[i,1]
        cz = cam_poses[i,2]
        

        # Transformation matrices (translation + rotations around x, y, z)
        mat_t_from[:,:,i] = np.array([[1,0,0,cx],
                             [0,1,0,cy],
                             [0,0,1,cz],
                             [0,0,0,1]])
        
        mat_t_to[:,:,i] = np.array([[1,0,0,-cx],
                             [0,1,0,-cy],
                             [0,0,1,-cz],
                             [0,0,0,1]])
        
        
    return np.array(mat_t_from), np.array(mat_t_to)
    
'''Rotation Matrices'''
def get_rotmats(cam_poses):
    
    mat_r_from = np.zeros((3,3,3))
    mat_r_to = np.zeros((3,3,3))
    for i in range(3): # Cam 1, 2, 3
        
        alpha = cam_poses[i,3]
        beta = cam_poses[i,4] 
        gamma = cam_poses[i,5]


        mat_rotx = np.array([[1,0,0],
                             [0,math.cos(alpha), -math.sin(alpha)],
                             [0, math.sin(alpha), math.cos(alpha)]])

        mat_roty = np.array([[math.cos(beta), 0, math.sin(beta)],
                             [0,1,0],
                             [-math.sin(beta), 0, math.cos(beta)]])


        mat_rotz = np.array([[math.cos(gamma), -math.sin(gamma), 0],
                             [math.sin(gamma), math.cos(gamma),0],
                             [0,0,1]])

        # General rotation matrix
        mat_r_from[:,:,i] = mat_rotz.dot(mat_roty).dot(mat_rotx)
        mat_r_to[:,:,i] = np.transpose(mat_r_from[:,:,i])
    
    
    return np.array(mat_r_from), np.array(mat_r_to)


'''
This functions determines the angular 'distance' between camera and object in planez XZ and YZ
'''
def get_angles_from_dvs(px, py, focl, cam_ix):

    angles = np.zeros(2)
    
    angles[0] = (180/math.pi)*math.atan2(px, focl[0,cam_ix]) 
    angles[1] = (180/math.pi)*math.atan2(py, focl[1,cam_ix]) 

    return angles

'''
This functions determines the angular 'distance' between camera and object in planez XZ and YZ
'''
def get_angles_from_pos(obj_pose):
    
    angles = np.zeros(2)
    
    angles[0] = (180/math.pi)*math.atan2((obj_pose[0]),(obj_pose[2])) + 180 # delta_x/delta_z
    angles[1] = (180/math.pi)*math.atan2((obj_pose[1]),(obj_pose[2])) + 180 # delta_y/delta_z

    if(angles[0]>180):
        angles[0] = 360-angles[0]
    if(angles[1]>180):
        angles[1] = 360-angles[1]
    if(angles[0]<-180):
        angles[0] = 360+angles[0]
    if(angles[1]<-180):
        angles[1] = 360+angles[1]

    if(obj_pose[0] < 0):
        angles[0] = -angles[0]
    if(obj_pose[1] < 0):
        angles[1] = -angles[1]

    return angles

'''  '''
def get_dvs_from_angles(angles, focl, cam_ix):

    px = math.tan((angles[0]*math.pi/180))*focl[0,cam_ix]
    py = math.tan((angles[1]*math.pi/180))*focl[1,cam_ix]

    return px, py

''' Create Multivariate Gaussian Distributions'''
def create_mgd(μ, Σ, r_v2r, r_r2w, t_r2w):   

    
    # t_r2w, t_w2r = get_transmats(cam_poses)
    # r_r2w, r_w2r = get_rotmats(cam_poses)
    

    r_μ = np.zeros((3,3))
    r_Σ = np.zeros((3,3,3))
    w_μ = np.zeros((3,3))
    w_Σ = np.zeros((3,3,3))
    new_μ = np.zeros((4,3)) # including a '1' at the end
    for k in range(3):
                                          
        # Rotating Means from virtual-cam space to real-cam space  
        r_μ[:,k] = r_v2r[:,:,k] @ μ
                 
        # Rotating Means from real-cam space to world space 
        w_μ[:,k] = r_r2w[:,:,k] @ r_μ[:,k]
    
        # Translating Means from Camera (Real=Virtual) space to World space 
        new_μ[:,k] = t_r2w[:,:, k] @ [w_μ[0,k], w_μ[1,k], w_μ[2,k],1]                     
                 
        # Rotating Covariance Matrix from virtual-cam space to real-cam space  
        r_Σ[:,:,k] = r_v2r[:,:,k] @ Σ @ r_v2r[:,:,k].T  
                 
        # Rotating Covariance Matrix from real-cam space to world space  
        w_Σ[:,:,k] = r_r2w[:,:,k] @ r_Σ[:,:,k] @ r_r2w[:,:,k].T 
    
    rv_1 = multivariate_normal(new_μ[0:3,0], w_Σ[:,:,0])
    rv_2 = multivariate_normal(new_μ[0:3,1], w_Σ[:,:,1])
    rv_3 = multivariate_normal(new_μ[0:3,2], w_Σ[:,:,2])
    
    return new_μ, w_Σ, [rv_1, rv_2, rv_3]


def analytical(μ, Σ):

    mu = np.zeros(3)
    V_n_p = np.zeros((3,3)) 
    
    V_1 = np.linalg.inv(Σ[:,:,0])
    V_n_p += V_1
    μ_1 = μ[0:3,0]

    V_2 = np.linalg.inv(Σ[:,:,1])
    V_n_p += V_2
    μ_2 = μ[0:3,1]

    V_3 = np.linalg.inv(Σ[:,:,2])
    V_n_p += V_3
    μ_3 = μ[0:3,2]

    V_n =np.linalg.inv(V_n_p)
    mu = ((V_1 @ μ_1) + (V_2 @ μ_2) + (V_3 @ μ_3)) @ V_n

    return mu