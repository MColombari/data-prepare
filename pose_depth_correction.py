import cv2
import numpy as np
import os
from tqdm import tqdm

def calculate_rotation_angle(joint1, joint2):

    delta_x = joint2[0] - joint1[0]
    delta_y = joint2[1] - joint1[1]
    delta_z = joint2[2] - joint1[2]
    
    angle_no_deepth = np.arctan2(delta_y, delta_x) * 180 / np.pi
    angle_with_deepth = np.arctan2(delta_z, delta_x) * 180 / np.pi
    angle_for_correction = np.arctan2(delta_z, delta_y) * 180 / np.pi

    if angle_no_deepth > 90  :
        angle_no_deepth -= 180
        
    if angle_with_deepth > 90:
        angle_with_deepth -= 180
    
    if angle_for_correction > 90:
        angle_for_correction -= 180
        
    if angle_no_deepth < -90  :
        angle_no_deepth += 180
        
    if angle_with_deepth < -90:
        angle_with_deepth += 180
    
    if angle_for_correction < -90:
        angle_for_correction += 180
    
    #print(f"angle_no_depth:{angle_no_deepth}, angle_with_depth={angle_with_deepth}, angle_for_correction:{angle_for_correction}")
    return angle_no_deepth, angle_with_deepth, angle_for_correction

def rotate_joints_3d(joints, angle_z, angle_y, center):
    angle_z_rad = np.deg2rad(angle_z)
    angle_y_rad = np.deg2rad(angle_y)

    rot_z = np.array([[np.cos(angle_z_rad), -np.sin(angle_z_rad), 0],
                  [np.sin(angle_z_rad), np.cos(angle_z_rad), 0],
                  [0, 0, 1]])

    rot_y = np.array([[np.cos(angle_y_rad), 0, np.sin(angle_y_rad)],
                      [0, 1, 0],
                      [-np.sin(angle_y_rad), 0, np.cos(angle_y_rad)]])

    rot_matrix = np.dot(rot_y, rot_z) # pre-moltiplication: rotations are wrt the base frame
    
    translated_joints = joints - np.array(center)
    rotated_joints = np.dot(translated_joints, rot_matrix.T)
    rotated_joints += np.array(center)
    
    return rotated_joints

def discretization(granularity,num_frames):
    
    num_bins= 180//granularity
    bins_angle= np.zeros((num_bins,3)) 
    angle_discretized = np.zeros(3)
    
    for k in range(num_frames):
        all_frames_angles[k] = calculate_rotation_angle(left_shoulder[k], right_shoulder[k])
        
        bin_no_depth = int(all_frames_angles[k][0] // granularity)  
        bin_with_depth = int(all_frames_angles[k][1] // granularity) 
        bin_for_correction = int(all_frames_angles[k][2] // granularity) 
        
        if bin_no_depth >= 0:
            bins_angle[bin_no_depth,0] += 1
        else: 
            bins_angle[bin_no_depth,0] += 1
            
        if bin_with_depth >= 0:
            bins_angle[bin_with_depth, 1] += 1
        else: 
            bins_angle[bin_with_depth, 1] += 1
            
        if bin_for_correction >= 0:
            bins_angle[bin_for_correction,2] += 1
        else: 
            bins_angle[bin_for_correction,2] += 1
    
    indices_max = np.argmax(bins_angle, axis=0)
    
    half = num_bins//2 
    if num_bins % 2 == 0:
        half -= 1
        
    for i in range(3):
        if 0 <= indices_max[i] <= (half):
            angle_discretized[i] = indices_max[i] * granularity
        else:
            angle_discretized[i] = (num_bins - indices_max[i]) * -granularity
            
    return angle_discretized

selected_joints = np.concatenate(([0, 5, 6, 7, 8, 9, 10], 
                    [91, 95, 96, 99, 100, 103, 104, 107, 108, 111], 
                    [112, 116, 117, 120, 121, 124, 125, 128, 129, 132]), axis=0)
shoulder_joints = [5, 6]

# Path
out_folder = '/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/data-prepare/demo/custom_dataset_depth_rotated_npy'
npy_folder = '/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/data-prepare/demo/custom_dataset_depth_npy'

for root, dirs, files in os.walk(npy_folder, topdown=False):
    for name in tqdm(files):
        npy = np.load(os.path.join(npy_folder, name)).astype(np.float32)
        npy = npy[:, :, :3]

        left_shoulder = npy[:, 5, :]
        right_shoulder = npy[:, 6, :]

        center_x = np.mean([left_shoulder[:, 0], right_shoulder[:, 0]], axis=0)
        center_y = np.mean([left_shoulder[:, 1], right_shoulder[:, 1]], axis=0)
        center_z = np.mean([left_shoulder[:, 2], right_shoulder[:, 2]], axis=0)

        new_npy_depth = npy.copy()
        
        correctionIndex = 1
        
        all_frames_angles= np.zeros((npy.shape[0],3))
        
        angle_discretized = np.zeros(3)
        
        #discretization, in this case with 5° of granularity
        angle_discretized = discretization(5, npy.shape[0]) #comment this line if you don't want to use discretized angles 
        
        for i in range(npy.shape[0]):
            
            delta_x= left_shoulder[i,0] - right_shoulder [i,0]
            delta_y = left_shoulder[i,1] - right_shoulder [i,1] 
            depth_pxl = left_shoulder[i,2] - right_shoulder [i,2] 
            
            #angle_no_depth, angle_with_depth, angle = calculate_rotation_angle(left_shoulder[i], right_shoulder[i]) #comment this line only if you want to use discretized angles

            #uncomment this following lines if you want to use fixed angles values:
            #if depth_pxl < 0:
            #    angle_with_depth = 45
            #elif depth_pxl > 0:
            #    angle_with_depth = -45
                
            center3D = (int(center_x[i]), int(center_y[i]),int(center_z[i]))
            rotated3D_shoulders = rotate_joints_3d(npy[i, :, :3], angle_discretized[0]  ,angle_discretized[1], center3D) #
            new_npy_depth[i, :, :] = rotated3D_shoulders

        npy = np.load(os.path.join(npy_folder, name)).astype(np.float32)
        new_npy_depth[:, :, 2] = npy[:, :, 3] #to create from 3D graph a 2D graph 
        np.save(os.path.join(out_folder, name[:-4] + '.npy'), new_npy_depth)
        
        
            
