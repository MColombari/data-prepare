import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL import Image

OUT_TEST_FOLDER = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/data-prepare/test_folder"

def calculate_rotation_angle(joint1, joint2):
    delta_x = joint2[0] - joint1[0]
    delta_y = joint2[1] - joint1[1]
    delta_z = joint2[2] - joint1[2]
    
    angle_no_deepth = np.arctan2(delta_y, delta_x) * 180 / np.pi
    angle_with_deepth = np.arctan2(delta_z, delta_x) * 180 / np.pi

    if angle_no_deepth > 90 or angle_with_deepth > 90:
        angle_no_deepth -= 180
        angle_with_deepth -= 180
    
    if angle_no_deepth < -90 or angle_with_deepth < -90:
        angle_no_deepth += 180
        angle_with_deepth += 180

    return angle_no_deepth, angle_with_deepth 

def rotate_joints_2d(joints, angle, center):
    angle_rad = np.deg2rad(angle)  # Convert angle to radians
    center_x, center_y = center

    # Calculate the cosine and sine of the angle
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Translate joints to origin
    translated_joints = joints - np.array(center)

    # Perform rotation
    rotated_joints_x = translated_joints[:, 0] * cos_angle - translated_joints[:, 1] * sin_angle
    rotated_joints_y = translated_joints[:, 0] * sin_angle + translated_joints[:, 1] * cos_angle

    # Translate joints back to original position
    rotated_joints = np.column_stack((rotated_joints_x + center_x, rotated_joints_y + center_y))

    return rotated_joints

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

def apply_translation_3d(joints, delta, axes=(0, 1, 2)):
    translation = np.zeros_like(joints)
    for axis, delta_value in zip(axes, delta):
        translation[:, axis] = delta_value
    return joints + translation

def calculate_distance_3d(joint, shoulder):
    return np.linalg.norm(joint - shoulder)

def apply_translation(joints, delta, axis=1):
    if joints.ndim == 1:
        translation = np.zeros_like(joints)
        translation[axis] = delta
        return joints + translation
    elif joints.ndim == 2:
        translation = np.zeros_like(joints)
        translation[:, axis] = delta
        return joints + translation
    else:
        raise ValueError("Unsupported dimension for joints array")

def calculate_distance(joint, shoulder):
    return np.linalg.norm(joint - shoulder)

def plot_skeleton_depth(npy, img, name_file):
    position_to_plot = [0, 5, 6, 9, 10]

    img = np.asarray(img)
    for pos in position_to_plot:
        x = 255 - npy[pos, 0]
        y = npy[pos, 1]
        img = cv2.circle(img, (int(x), int(y)), radius=int(20 * npy[pos, 2]), color=(255, 0, 0), thickness=0)
    img = Image.fromarray(img)
    img.save(f'{OUT_TEST_FOLDER}/{name_file}.png')

selected_joints = np.concatenate(([0, 5, 6, 7, 8, 9, 10], 
                    [91, 95, 96, 99, 100, 103, 104, 107, 108, 111], 
                    [112, 116, 117, 120, 121, 124, 125, 128, 129, 132]), axis=0)
shoulder_joints = [5, 6]

# Path
out_folder = '/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/data-prepare/demo/custom_dataset_rotated_npy'
npy_folder = '/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/data-prepare/demo/custom_dataset_npy'

for root, dirs, files in os.walk(npy_folder, topdown=False):
    for name in tqdm(files):
        npy = np.load(os.path.join(npy_folder, name)).astype(np.float32)
        npy = npy[:, selected_joints, :3]

        left_shoulder = npy[:, 5, :]
        right_shoulder = npy[:, 6, :]

        center_x = np.mean([left_shoulder[:, 0], right_shoulder[:, 0]], axis=0)
        center_y = np.mean([left_shoulder[:, 1], right_shoulder[:, 1]], axis=0)
        center_z = np.mean([left_shoulder[:, 2], right_shoulder[:, 2]], axis=0)

        new_npy_depth = npy.copy()
        new_npy_no_depth = npy.copy()
        
        for i in range(npy.shape[0]):
            depth_pxl = left_shoulder[i][2] - right_shoulder [i][2]
            angle_no_depth, angle_with_depth = calculate_rotation_angle(left_shoulder[i], right_shoulder[i])
            #fai print angolo con depth
            delta_x= left_shoulder[i][0] - right_shoulder [i][0]
            #calibration step (depth_pxl= k* depth, con depth=ipotenusa*sin(alfa) con ipotenusa che trovo come distanza 3d )
            #printa quelli per cui non è rispettata questa cosa
            
            
            
            center3D = (int(center_x[i]), int(center_y[i]),int(center_z[i]))
            rotated3D_shoulders = rotate_joints_3d(npy[i, shoulder_joints, :], angle_no_depth ,angle_with_depth, center3D) #ruoto anche rispetto alla profondità
            new_npy_depth[i, shoulder_joints, :] = rotated3D_shoulders
            
            # Calcola la traduzione per i giunti vicini
            original_shoulder_y = np.mean(npy[i, shoulder_joints, 1])
            rotated_shoulder_y = np.mean(rotated3D_shoulders[:, 1])
            delta_y = rotated_shoulder_y - original_shoulder_y

            # Determina la vicinanza degli altri giunti alla spalla destra
            distances_to_right_shoulder = [calculate_distance_3d(npy[i, j, :], right_shoulder[i]) for j in range(npy.shape[1])]
            distances_to_left_shoulder = [calculate_distance_3d(npy[i, j, :], left_shoulder[i]) for j in range(npy.shape[1])]
            proximity_indices = [idx for idx in range(npy.shape[1]) if distances_to_right_shoulder[idx] < distances_to_left_shoulder[idx]]

            # Applica la traduzione ai giunti più vicini alla spalla destra
            for joint_idx in proximity_indices:
                new_npy_depth[i, joint_idx, :] = apply_translation_3d(npy[i, joint_idx, :], (0, delta_y, 0))


        np.save(os.path.join(out_folder, name[:-4] + '.npy'), new_npy_depth)
