import csv
from toolsdsgen import *

class Cameras:
  def __init__(self, cam_poses, focl, w, h):
    self.cam_poses = cam_poses
    self.focl = focl
    self.w = w
    self.h = h

def gen_data(cams, μ_w):
    
    
    
    t_r2w, t_w2r = get_transmats(cams.cam_poses)
    r_r2w, r_w2r = get_rotmats(cams.cam_poses)


    aux_a = np.zeros((4,3))
    aux_b = np.zeros((3,3))

    μ_r = np.zeros((3,3)) 
    μ_f = np.zeros((4,4)) 

    cam_ang_fp = np.zeros((2,3))
    px = np.zeros(3)
    py = np.zeros(3)
    snn_x = np.zeros(3)
    snn_y = np.zeros(3) 


    for k in range(3):


        aux_a[:,k] = t_w2r[:,:, k] @ np.concatenate((μ_w,[1]))
        μ_r[:,k]  = r_w2r[:,:,k] @ aux_a[0:3,k]

        # Going Back to World Space from Real-Camera Space
        aux_b[:,k] = r_r2w[:,:,k] @ μ_r[0:3,k]
        μ_f[:,k] = t_r2w[:,:, k] @ np.concatenate((aux_b[:,k],[1]))

        # Estimating Which pixels are concerned
        cam_ang_fp[:,k] = get_angles_from_pos(μ_r[:,k])
        px[k], py[k] = get_dvs_from_angles(cam_ang_fp[:,k], cams.focl, k)
        snn_x[k] = px[k]/(cams.w/2)
        snn_y[k] = py[k]/(cams.h/2)
    
    

#     print("Poses in Real-Camera Space")
#     print(μ_r[:,0])
#     print(μ_r[:,1])
#     print(μ_r[:,2])

#     print("Coming Back to World Space")
#     print(μ_f[0:3,0])
#     print(μ_f[0:3,1])
#     print(μ_f[0:3,2])

#     print("Angles")
#     print(cam_ang_fp[:,0])
#     print(cam_ang_fp[:,1])
#     print(cam_ang_fp[:,2])


#     print("Pixels")

#     print(f"Cam 1: x:{px[0]} y:{py[0]}")
#     print(f"Cam 2: x:{px[1]} y:{py[1]}")
#     print(f"Cam 3: x:{px[2]} y:{py[2]}")

#     print("SNN")

#     print(f"Cam 1: x:{snn_x[0]} y:{snn_y[0]}")
#     print(f"Cam 2: x:{snn_x[1]} y:{snn_y[1]}")
#     print(f"Cam 3: x:{snn_x[2]} y:{snn_y[2]}")
    
    return snn_x, snn_y
    

cam_poses = set_cam_poses()
focl = set_focal_lengths()
cams = Cameras(cam_poses, focl, 640, 480)

# Object pose in world space
μ_w = np.array([-0.25, 1.00, 0.50])


nb_pts = 40
x_array = np.linspace(-1.0,0.0,nb_pts)
y_array = np.linspace(0.0,1.0,nb_pts)
z_array = np.linspace(0.5,1.5,nb_pts)

count = 0
with open('dataset.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    for x in x_array:
        for y in y_array:
            for z in z_array:
                μ_w = np.array([x, y, z])
                snn_x, snn_y = gen_data(cams, μ_w)
                if abs(snn_x[0])<=1 and abs(snn_x[1])<=1 and abs(snn_x[2])<=1:
                    if abs(snn_y[0])<=1 and abs(snn_y[1])<=1 and abs(snn_y[2])<=1:
                        line = [x,y,z,snn_x[0],snn_y[0],snn_x[1],snn_y[1],snn_x[2],snn_y[2]]
                        writer.writerow(line)
                        count += 1
print(f"Count = {count}/{nb_pts**3}")