import cv2
import numpy as np
import math
import os
from tqdm import tqdm
from wholepose.utils import plot_31_pose

from PIL import Image  

OUT_TEST_FOLDER = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/data-prepare/test_folder"


def calculate_rotation_angle(joint1, joint2):
    delta_x = joint2[0] - joint1[0]
    delta_y = joint2[1] - joint1[1]
    delta_z = joint2[2] - joint1[2]
    
    angle_no_deepth = np.arctan2(delta_y, delta_x) * 180/np.pi # angle without considering depth
    angle_with_deepth = np.arctan2(delta_z, delta_x) * 180 / np.pi  # angle in plane xz
    #angle_yz = np.arctan2(delta_z, delta_y) * 180 / np.pi  # angle in plane yz
    #angle_with_deepth = np.mean([angle_xz, angle_yz]) # combination of deepth angles
    
    if angle_no_deepth > 90 or angle_with_deepth > 90:
        angle_no_deepth -= 180
       # angle_with_deepth -= 180
    
    if angle_no_deepth < -90 or angle_with_deepth < -90:
        angle_no_deepth += 180
       # angle_with_deepth += 180

    return angle_no_deepth, angle_with_deepth


def joint_rotation(joints, angle, center, type):

    # joints's rotation
    ones = np.ones(shape=(joints.shape[0], 1))
    joints_homogeneous = np.hstack([joints, ones])
    
    if type == "no_deepth":
        M = cv2.getRotationMatrix2D((center[0],center[1]), angle, 1.0)
        rotated_joints = M.dot(joints_homogeneous.T).T
        
    elif type == "with_deepth":
        # joints' 3D rotation (considering deepth)
        M_xz = cv2.getRotationMatrix2D((center[0], center[2]), angle, 1.0)
        #M_yz = cv2.getRotationMatrix2D((center[1], center[2]), angle, 1.0)
        # joints' rotation
        rotated_joints = M_xz.dot(joints_homogeneous[:, [0, 2]].T).T

        #rotated_joints_yz = np.dot(joints_homogeneous[:, [1, 2]], M_yz[:, :2].T)
        #rotated_joints = np.column_stack((rotated_joints_xz[:, 0], rotated_joints_yz[:, 0], rotated_joints_xz[:, 1]))
    
    return rotated_joints

def plot_skeleton_depth(npy, img, name_file):
    position_to_plot = [0,5,6, 9, 10]

    img = np.asarray(img)
    print(img.shape)
    for pos in position_to_plot:
        x = 255 - npy[pos, 0]
        y = npy[pos, 1]
        print(f"x: {x}, y: {y}, z:{npy[pos, 2]}, pos: {pos}")
        img = cv2.circle(img, (int(x), int(y)), radius=int(20*npy[pos, 2]), color=(255,0,0), thickness=0)
    img = Image.fromarray(img)
    img.save(f'{OUT_TEST_FOLDER}/{name_file}.png')
    # cv2.imwrite('OUT_TEST_FOLDER/{}.png'.format('test'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


selected_joints = np.concatenate(([0,5,6,7,8,9,10], 
                    [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0)
shoulder_joints=[5,6]

# Path
out_folder = '/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/data-prepare/rotated_test_npy'
npy_folder='/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/data-prepare/demo/test_npy'

for root, dirs, files in os.walk(npy_folder, topdown=False):
    for name in tqdm(files):
        npy = np.load(os.path.join(npy_folder, name)).astype(np.float32)
        npy = npy[:, selected_joints, :3] 
        #print(f"old_npy_shape: {npy.shape}")
        
        # shoulders' joints selection
        left_shoulder = npy[:, 5, :]
        right_shoulder = npy[:, 6, :]

        center_x = np.mean([left_shoulder[:,0],right_shoulder[:,0]], axis=0)
        center_y = np.mean([left_shoulder[:,1],right_shoulder[:,1]], axis=0)
        center_z = np.mean([left_shoulder[:,2],right_shoulder[:,2]], axis=0)
        #print(f'left_sh: {left_shoulder[1]}, right_sh: {right_shoulder[1]}')
        #print(f'center_x: {center_x[1]}, center_y: {center_y[1]}, center_z: {center_z[1]}')

        new_npy_no_deepth=np.zeros(npy.shape)

        for i in range(npy.shape[0]):
            # Calculate rotation angle
            angle_no_deepth, angle_with_deepth = calculate_rotation_angle(left_shoulder[i], right_shoulder[i])
            
            #if i == 1:
            #   print(f'angle_no_deepth:{angle_no_deepth}, angle_with_deepth:{angle_with_deepth}')
            
            new_npy_no_deepth[i,:,:2] = joint_rotation(npy[i,:,:2], angle_no_deepth, [int(center_x[i]), int(center_y[i])], type="no_deepth")
            #new_npy_with_deepth[i,:,:] = rotate_image_and_joints(npy[i,:,:], angle_with_deepth, [int(center_x[i]), int(center_y[i]), int(center_z[i])], type="with_deepth")
        #print(f"new_npy_shape: {new_npy_no_deepth.shape}")
        #exit()
        #save
        #np.save(os.path.join(out_folder, name[:-4] + '_rotated.npy'), new_npy_no_deepth)
        np.save(os.path.join(out_folder, name[:-4] + '.npy'), new_npy_no_deepth) #perchè se no scazza tutto dopo 
        