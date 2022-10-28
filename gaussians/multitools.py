
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import multivariate_normal
import time
import math
from scipy import stats
import time
import math
import matplotlib.pyplot as plt
import datetime

# from visuals import visualize_cigar
# from utils import gen_rnd_cov

def set_cam_poses():

    cam_poses = np.zeros((3,6)) # 3 cameras, 6 parameters
    
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

def set_vir_poses(angles):
       
    vir_poses = np.zeros((3,6))

    # Cam 1
    vir_poses[0,0] = 0
    vir_poses[0,1] = 0
    vir_poses[0,2] = 0
    vir_poses[0,3] = (math.pi/180)*(angles[1,0])    # around X axis -->  ang(YZ)
    vir_poses[0,4] = (math.pi/180)*(-angles[0,0])   # around Y axis --> -ang(XZ)
    vir_poses[0,5] = 0 

    # Cam 2
    vir_poses[1,0] = 0
    vir_poses[1,1] = 0
    vir_poses[1,2] = 0
    vir_poses[1,3] = (math.pi/180)*(angles[1,1])    # around X axis -->  ang(YZ)
    vir_poses[1,4] = (math.pi/180)*(-angles[0,1])   # around Y axis --> -ang(XZ)
    vir_poses[1,5] = 0

    # Cam 3
    vir_poses[2,0] = 0
    vir_poses[2,1] = 0
    vir_poses[2,2] = 0
    vir_poses[2,3] = (math.pi/180)*(angles[1,2])    # around X axis -->  ang(YZ)
    vir_poses[2,4] = (math.pi/180)*(-angles[0,2])   # around Y axis --> -ang(XZ)
    vir_poses[2,5] = 0 

    
    
    return vir_poses

''' Translation Matrices'''
def get_transmats(cam_poses):
    
    mat_tran = np.zeros((4,4,3))
    for i in range(3): # Cam 1, 2, 3
        
        cx = cam_poses[i,0]
        cy = cam_poses[i,1]
        cz = cam_poses[i,2]

        # Transformation matrices (translation + rotations around x, y, z)
        mat_tran[:,:,i] = np.array([[1,0,0,cx],
                             [0,1,0,cy],
                             [0,0,1,cz],
                             [0,0,0,1]])
        
    return mat_tran
    
    
    
'''Rotation Matrices'''
def get_rotmats(cam_poses):
    
    mat_rota = np.zeros((3,3,3))
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
        mat_rota[:,:,i] = mat_rotz.dot(mat_roty).dot(mat_rotx)
    
    
    return mat_rota

'''
This function defines object pose in B space starting from A space
It performs a full coordinate transformation (rotation and translation)
b2a is rotation matrix (3x3)
trl is translation matrix (4x4)
'''
def get_b_pose_from_a_pose(b2a, trl, a_pose):
        
    mat = np.zeros((4,4))
    b_pose = np.zeros(4) # coordinates|cameras
            
    mat [3,:] = trl[3,:]
    mat [:,3] = trl[:,3]
    mat[0:3, 0:3] = b2a[0:3,0:3]
        
    a2b = np.linalg.inv(mat)
    b_pose = a2b.dot(np.array([a_pose[0], a_pose[1], a_pose[2], 1]))
    
#     print("b_pose")
#     print(np.round(b_pose[0:3],3))

    return b_pose[0:3]

def get_angles(obj_pose):
    
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

        

def get_joint_rv(x, y, z, rv):
    
    p = rv.pdf([x, y, z])
    
        
    return p
 
def get_3_joint_rv(x, y, z, rv):
    
    p1 = rv[0].pdf([x, y, z])
    p2 = rv[1].pdf([x, y, z])
    p3 = rv[2].pdf([x, y, z])
    
    p = np.cbrt(p1*p2*p3)
    
#     p = p1*p2*p3
        
    return p

    
def visualize_intersection(xx, yy, zz, p, idx, w_pose):
        
    
    # Creating figure
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection="3d")
    

    # Creating plot
    ax.scatter3D(xx[idx],yy[idx], zz[idx], c=p[idx], cmap='viridis', vmin=0, vmax=10, marker='.')
    ax.scatter3D(w_pose[0], w_pose[1], w_pose[2], cmap='Reds', vmin=0, vmax=10, marker='p')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')    
    
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    
#     ax.view_init(0,180) # Top View
    
    plt.show()  

    

def get_euclidian_d(w_pose, prediction):
    dx = abs(w_pose[0]-prediction[0])
    dy = abs(w_pose[1]-prediction[1])
    dz = abs(w_pose[2]-prediction[2])
    
    print("dx = {:.3f} | dy = {:.3f} | dz = {:.3f} in cm".format(100*dx, 100*dy, 100*dz))
    
    d = math.sqrt(dx*dx+dy*dy+dz*dz)
    return d
    
''' This function prepares the ground for coordinate transformation'''
def prepare_ground(w_pose):
        
    # Get camera poses
    r_cam_poses = set_cam_poses()
    
    # Get Rotation Matrices: real-cam to world 
    r2w = get_rotmats(r_cam_poses)
    
    # Get Translation Matrices: real-cam to world
    r_trl = get_transmats(r_cam_poses)
        
    # Get poses and angles of connecting ray vs Z axis in real-cam space
    r_obj_poses = np.zeros((3,3))       
    r_obj_angles = np.zeros((2,3))    
    for k in range(3):
        r_obj_poses[:,k] = get_b_pose_from_a_pose(r2w[:,:,k], r_trl[:,:,k], w_pose)
        r_obj_angles[0:2, k] = get_angles(r_obj_poses[:,k])
            
    # Estimate virtual camera poses (in real camera space)
    v_poses = set_vir_poses(r_obj_angles)
        
    # Get Rotation Matrices: virtual-cam to real-cam 
    v2r = get_rotmats(v_poses)
        
    # Get poses in virtual-cam space
    v_obj_poses = np.zeros((3,3))  
    v_obj_angles = np.zeros((2,3))    
    for k in range(3):
        v_obj_poses[:,k] = get_b_pose_from_a_pose(v2r[:,:,k], np.identity(4), r_obj_poses[:,k])
        v_obj_angles[0:2, k] = get_angles(v_obj_poses[:,k])
        # @TODO: The angles here should be exactly ZERO !!! WTF?!?!?
#         print(np.round(v_poses[:,k],3))        
#         print(v_angles[0:2, k])
    print("vir_z_123 = ({:.3f}, {:.3f}, {:.3f})".format(np.round(v_obj_poses[2,0],3), np.round(v_obj_poses[2,1],3), np.round(v_obj_poses[2,2],3)))
    
    return r_trl, r2w, v2r, r_obj_poses, v_obj_poses, r_cam_poses

''' Create Multivariate Gaussian Distributions'''
def create_mgd(μ, Σ, r_trl, r2w, v2r, v_obj_poses):   
   
    r_μ = np.zeros((3,3))
    r_Σ = np.zeros((3,3,3))
    w_μ = np.zeros((3,3))
    w_Σ = np.zeros((3,3,3))
    new_μ = np.zeros((4,3)) # including a '1' at the end
    for k in range(3):
        
        # @TODO: only for testing purposes
#         μ[2] = v_obj_poses[2,k]
                                      
        # Rotating Means from virtual-cam space to real-cam space  
        r_μ[:,k] = v2r[:,:,k] @ μ
                 
        # Rotating Means from real-cam space to world space 
        w_μ[:,k] = r2w[:,:,k] @ r_μ[:,k]
    
        # Translating Means from Camera (Real=Virtual) space to World space 
        new_μ[:,k] = r_trl[:,:, k] @ [w_μ[0,k], w_μ[1,k], w_μ[2,k],1]                     
                 
        # Rotating Covariance Matrix from virtual-cam space to real-cam space  
        r_Σ[:,:,k] = v2r[:,:,k] @ Σ @ v2r[:,:,k].T  
                 
        # Rotating Covariance Matrix from real-cam space to world space  
        w_Σ[:,:,k] = r2w[:,:,k] @ r_Σ[:,:,k] @ r2w[:,:,k].T 
    
    rv_1 = multivariate_normal(new_μ[0:3,0], w_Σ[:,:,0])
    rv_2 = multivariate_normal(new_μ[0:3,1], w_Σ[:,:,1])
    rv_3 = multivariate_normal(new_μ[0:3,2], w_Σ[:,:,2])
    
    return new_μ, [rv_1, rv_2, rv_3]

''' This functions creates linspaces for (x, y, z) used as w_pose '''
def get_w_bases(nb_samples):
    
    w_x = np.linspace(-0.5, 0, num=nb_samples) 
    w_y = np.linspace(0.0, 1, num=nb_samples)
    w_z = np.linspace(0.7, 1.2, num=nb_samples)
    
    return w_x, w_y, w_z

    
def predict_pose(nb_pts, rv, mean, w_pose):
    
        
    diff = 1.0
    x_0 = np.mean(mean[0,:])
    y_0 = np.mean(mean[1,:])
    z_0 = np.mean(mean[2,:])
#     lims = np.array([[-1, 3],[-1, 3],[-1,3]])
#     lims = np.array([[-1.5, 1.5],[-1.5, 1.5],[-1.5, 1.5]])
    lims = np.array([[x_0-diff, x_0+diff],[y_0-diff, y_0+diff],[z_0-diff, z_0+diff]])
    nzc_max = 100 
    th_increase = 0.001
    count = 0
    while True:
        x = np.linspace(lims[0,0], lims[0,1], num=nb_pts) 
        y = np.linspace(lims[1,0], lims[1,1], num=nb_pts)
        z = np.linspace(lims[2,0], lims[2,1], num=nb_pts)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
        xyz = np.zeros((nb_pts,nb_pts, nb_pts,3))
        xyz[:,:,:,0] = xx
        xyz[:,:,:,1] = yy
        xyz[:,:,:,2] = zz

        # Getting joint probabilities
        start = datetime.datetime.now()

        p1 = rv[0].pdf(xyz)
        p2 = rv[1].pdf(xyz)
        p3 = rv[2].pdf(xyz)

    #     p = np.cbrt(np.multiply(p1,np.multiply(p2,p3)))
        p = np.cbrt(p1*p2*p3)
        # p = p/np.sum(p)


        stop = datetime.datetime.now()
        elapsed = stop - start

        threshold = 0
        while True:
            idx = p > threshold
            nzc = np.count_nonzero(idx)
            threshold += th_increase
#             print("th={:.3f} --> ## ={:.3f}".format(threshold, nzc))
            if nzc < nzc_max:
                break

        # Indices of Max Probability
        imp = np.unravel_index(np.argmax(p, axis=None), p.shape) 
        prediction = [x[imp[0]], y[imp[1]], z[imp[2]]]

        
        count+= 1     
        nzc_max = 20*nzc_max
        th_increase = 10*th_increase
        delta = 0.05
        lims = np.array([[x[imp[0]]-delta, x[imp[0]]+delta],[y[imp[1]]-delta, y[imp[1]]+delta],[z[imp[2]]-delta, z[imp[2]]+delta]])
        if count > 1:
            print("Joint probabilities obtained after: " + str(int(elapsed.microseconds/1000)) + " [ms].")
            print("Final Threshold : {:.3f}".format(threshold))
            print("Prediction: ({:.3f}, {:.3f}, {:.3f})".format(x[imp[0]], y[imp[1]], z[imp[2]]))
            print("Probability: {:.3f}".format(np.argmax(p, axis=None)))
            d = get_euclidian_d(w_pose, prediction)
            print("\033[1m" + "Error: {:.3f} [cm]\n\n".format(100*d)+ "\033[0m")
            break
    
    return xx, yy, zz, p, idx, prediction, d
  
def visualize_all_cigars(nb_pts, rv, mean, plane, r_cam_poses):
    
    nzc_max =  1000
    th_increase = 0.001
    
    # Creating figure
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection="3d")
    cmappers = ['Greens', 'Blues', 'Reds']
   
    # 'Plot' workspace
    xs = np.linspace(-0.7, 0, nb_pts)
    ys = np.linspace(-0.1, 1, nb_pts)
    zs = np.linspace(0, 1.6, nb_pts)

    X, Y = np.meshgrid(xs, ys, indexing='ij')
    Z = np.ones((nb_pts,nb_pts))*1.5
    ax.plot_surface(X, Y, Z, alpha=0.2, color='k')

    Y, Z = np.meshgrid(ys, zs, indexing='ij')
    X = np.ones((nb_pts,nb_pts))*-0.6
    ax.plot_surface(X, Y, Z, alpha=0.2, color='k')

    X, Z = np.meshgrid(xs, zs, indexing='ij')
    Y = np.ones((nb_pts,nb_pts))*0
    ax.plot_surface(X, Y, Z, alpha=0.2, color='k')
    
    # Plot Gaussians
    p = np.zeros((nb_pts, nb_pts, nb_pts))
    for k in range(3):
#         lims = np.array([[-1, 3],[-1, 3],[-1,3]])
        diff = 1.0
        lims = np.array([[mean[0,k]-diff, mean[0,k]+diff],[mean[1,k]-diff, mean[1,k]+diff],[mean[2,k]-diff, mean[2,k]+diff]])
        x = np.linspace(lims[0,0], lims[0,1], num=nb_pts) 
        y = np.linspace(lims[1,0], lims[1,1], num=nb_pts)
        z = np.linspace(lims[2,0], lims[2,1], num=nb_pts)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

        xyz = np.zeros((nb_pts,nb_pts, nb_pts,3))
        xyz[:,:,:,0] = xx
        xyz[:,:,:,1] = yy
        xyz[:,:,:,2] = zz
    
        
        
        start = time.time()
        
        p=rv[k].pdf(xyz)

        stop = time.time()
        elapsed = stop - start
        print("Joint probabilities obtained after: " + str(int(elapsed)) + " seconds.")

        threshold = 0
        while True:
            idx = p > threshold
            nzc = np.count_nonzero(idx)
            threshold += th_increase
#             print("th={:.3f} --> ## ={:.3f}".format(threshold, nzc))
            if nzc < nzc_max:
                break

        # Creating plot
        ax.scatter3D(xx[idx], yy[idx], zz[idx], c=p[idx], cmap=cmappers[k], vmin=0, vmax=10, marker='.')
    
    cam_colors = ['g', 'b', 'r']
    # Plot Cameras    
    for k in range(3):
        ax.scatter3D(r_cam_poses[k,0], r_cam_poses[k,1], r_cam_poses[k,2], vmin=0, vmax=10, marker='p', color=cam_colors[k])
    
    # Plot Origin
    ax.scatter3D(0, 0, 0, vmin=0, vmax=10, marker='p', color='k')
    

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z') 
    
    
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.0, 2.0])
    ax.set_zlim([-0.5, 2.5])
    
    if plane == 'xz':
        ax.view_init(0, 90) 
    if plane == 'xy':
        ax.view_init(-90, 90)
    if plane == 'yz':
        ax.view_init(0, 0) 
    if plane == 'im':
        ax.set_xlim([-1, 0.5])
        ax.set_ylim([-0.5, 1.0])
        ax.set_zlim([0, 1.5])
        ax.view_init(-30, 10)     
        plt.savefig('Example.png', bbox_inches='tight',pad_inches = 0)

    plt.show()  