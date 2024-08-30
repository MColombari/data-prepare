import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from scipy.stats import mode

OUT_TEST_FOLDER = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/data-prepare/test_folder"

def calculate_rotation_angle(joint1, joint2):

    delta_x = joint2[0] - joint1[0]
    delta_y = joint2[1] - joint1[1]
    delta_z = joint2[2] - joint1[2]
    #print(f"punto1_xyz: {joint1[0]}, {joint1[1]}, {joint1[2]}")
    #print(f"punto2_xyz: {joint2[0]}, {joint2[1]}, {joint2[2]}")
    
    #print(f"delta_x:{delta_x}, delta_y={delta_y}, delta_z:{delta_z}")
    
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

def rotate_joints_3d(joints, angle_x, angle_y, center):
    angle_x_rad = np.deg2rad(angle_x)
    angle_y_rad = np.deg2rad(angle_y)

    # Matrici di rotazione 3D
    rot_x = np.array([[1, 0, 0],
                      [0, np.cos(angle_x_rad), -np.sin(angle_x_rad)],
                      [0, np.sin(angle_x_rad), np.cos(angle_x_rad)]])

    rot_y = np.array([[np.cos(angle_y_rad), 0, np.sin(angle_y_rad)],
                      [0, 1, 0],
                      [-np.sin(angle_y_rad), 0, np.cos(angle_y_rad)]])

    # Rotazione combinata intorno agli assi X e Y
    rot_matrix = np.dot(rot_y, rot_x)

    # Traslazione giunti al centro
    translated_joints = joints - np.array(center)

    # Rotazione dei giunti
    rotated_joints = np.dot(translated_joints, rot_matrix.T)

    # Traslazione dei giunti alla posizione originale
    rotated_joints += np.array(center)
    
    return rotated_joints

def calculate_distance_3d(joint, shoulder):
    return np.linalg.norm(joint - shoulder)

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
        print(f"\n{name}")
        
        all_frames_angles= np.zeros((npy.shape[0],3))
        
        #angle_video = np.zeros(3)  #--> per contenere i valori ottenuti con la moda
        #median_angle = np.zeros(3) #--> per contenere i valori ottenuti con la mediana
        
        angle_discretized = np.zeros(3)
        
        bins_angle= np.zeros((18,3)) #ho 18 bin da 10 gradi ciascuno
        
        #salva per tutti i frame i bin incrementati in base all'angolo
        for k in range(npy.shape[0]):
            all_frames_angles[k] = calculate_rotation_angle(left_shoulder[k], right_shoulder[k])
            
            bin_no_depth = int(all_frames_angles[k][0] // 10)  #primo angolo
            bin_with_depth = int(all_frames_angles[k][1] // 10) #secondo angolo
            bin_for_correction = int(all_frames_angles[k][2] // 10) #terzo angolo
            
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
        
        for i in range(3):
            if 0 <= indices_max[i] <= 8:
                angle_discretized[i] = indices_max[i] * 10
            else:
                angle_discretized[i] = (18 - indices_max[i]) * -10

        #angle_video[:]= mode(all_frames_angles, axis=0,  keepdims=False).mode[0] #-->trova i valori usando la moda
        #median_angle[:]= np.median(all_frames_angles, axis=0) #--> trova i valori usando la mediana
        
        delta_x = left_shoulder[1][0] - right_shoulder[1][0]
        delta_y = left_shoulder[1][1] - right_shoulder[1][1]
        delta_z = left_shoulder[1][2] - right_shoulder[1][2]
        print(f"Right Shoulder coord: {right_shoulder[1][0]}, {right_shoulder[1][1]}, {right_shoulder[1][2]}")
        print(f"Left Shoulder coord: {left_shoulder[1][0]}, {left_shoulder[1][1]}, {left_shoulder[1][2]}")
    
        print(f"delta_x:{delta_x}, delta_y={delta_y}, delta_z:{delta_z}")
        print(f"angle discretized: {angle_discretized}")
        #print(f"Frequent angle value in video (Moda):{angle_video}")
        #print(f"Central angle values in video (Mediana):{median_angle}")
        
        for i in range(0):
            
            depth_pxl = left_shoulder[i,2] - right_shoulder [i,2] # è delta z
            delta_x= left_shoulder[i,0] - right_shoulder [i,0]
            delta_y = left_shoulder[i,1] - right_shoulder [i,1] 
            
            angle_no_depth, angle_with_depth, angle = calculate_rotation_angle(left_shoulder[i], right_shoulder[i])
            
            #epsilon = 1e-5  # Tolleranza per il confronto
            #print(angle_with_depth)
            #if not (np.isclose(angle_with_depth, 180.0, atol=epsilon) or np.isclose(angle_with_depth, 0.0, atol=epsilon)):
            #    print(angle_with_depth)
            #    exit() 
            
            #print(correctionIndex)
            #calibration step (depth_pxl= k* depth, con depth=ipotenusa*sin(alfa) con ipotenusa che trovo come distanza 3d )
            dist3D = calculate_distance_3d(left_shoulder[i], right_shoulder[i])
            
            #print(dist3D)
            left_shoulder_xz = np.array([left_shoulder[i][0], 0, left_shoulder[i][2]], dtype=np.float32)
            right_shoulder_xz = np.array([right_shoulder[i][0], 0, right_shoulder[i][2]], dtype=np.float32)
            distXZ = calculate_distance_3d(left_shoulder_xz, right_shoulder_xz)
            
            left_shoulder_xy = np.array([left_shoulder[i][0], left_shoulder[i][1], 0], dtype=np.float32)
            right_shoulder_xy = np.array([right_shoulder[i][0], right_shoulder[i][1], 0], dtype=np.float32)
            distXY = calculate_distance_3d(left_shoulder_xy, right_shoulder_xy)

            #print(distXY)
            condition = (depth_pxl == (correctionIndex * np.sin(angle_with_depth)*distXZ))
            #print(correctionIndex * np.sin(angle_with_depth)*distXY)
            #print(depth_pxl)
            
            
            if depth_pxl and not (condition):
                #print("dentro")
                #print(angle)
                #print(angle_with_depth)
                angle_with_depth = angle
                correctionIndex = depth_pxl / (np.sin(angle_with_depth) * distXY)
                #print(correctionIndex)

            #print(angle)
            #print(angle_with_depth)
            #correctionIndex = depth_pxl / np.sin(angle)* dist3D
            
            #printa quelli per cui non è rispettata questa cosa
            
            center3D = (int(center_x[i]), int(center_y[i]),int(center_z[i]))
            rotated3D_shoulders = rotate_joints_3d(npy[i, shoulder_joints, :3], angle_no_depth ,angle_with_depth, center3D) #ruoto anche rispetto alla profondità
            new_npy_depth[i, shoulder_joints, :] = rotated3D_shoulders
            
            #calcolo distanza sul piano xy prima della rotazione e post rotazione e se è diversa aggiusto le coordinate di un fattore moltiplicativo
            
            xy_rotated_distance = new_npy_depth[i, 5 , 1] - new_npy_depth[i, 6 , 1]
            

        npy = np.load(os.path.join(npy_folder, name)).astype(np.float32)
        new_npy_depth[:, :, 2] = npy[:, :, 3]
            
        #np.save(os.path.join(out_folder, name[:-4] + '.npy'), new_npy_depth)
        
        
            
