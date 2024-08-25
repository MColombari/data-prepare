import os
import numpy as np
import cv2
from tqdm import tqdm
import math

OUT_TEST_FOLDER = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/data-prepare/test_folder"

def calculate_rotation_angle(joint1, joint2):
    #calcola l'angolo di rotazione
    delta_x = joint2[0] - joint1[0]
    delta_y = joint2[1] - joint1[1]
    delta_z = joint2[2] - joint1[2]

    theta = math.atan2(delta_y, delta_x)
    
    # Matrice di rotazione
    R = np.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta), math.cos(theta), 0],
        [0, 0, 1]
    ])
    
    # Matrice di traslazione per spostare il segmento all'altezza desiderata
    y_new = (joint1[1] + joint2[1]) / 2
    T = np.array([
        [1, 0, -((joint1[0] + joint2[0]) / 2)],
        [0, 1, -y_new],
        [0, 0, 1]
    ])
    
    # Matrice combinata di trasformazione
    M = T @ R
    
    # Vettori dei punti originali
    P1 = np.array([joint1[0], joint2[1], 1])
    P2 = np.array([joint2[0], joint2[1], 1])
    
    # Applicazione della trasformazione
    P1_transformed = M @ P1
    P2_transformed = M @ P2
    
    # Estrazione delle coordinate trasformate
    #x1_new, y1_new = P1_transformed[0], P1_transformed[1]
    #x2_new, y2_new = P2_transformed[0], P2_transformed[1]
    print(f"delta_x: {delta_x}, delta_y: {delta_y}, delta_z:{delta_z}")
    print(M)
    return (M @ P1), (M @ P2), M
    
    angle_no_depth = np.arctan2(delta_y, delta_x) * 180 / np.pi  # angle without considering depth
    angle_with_depth = np.arctan2(delta_z, delta_x) * 180 / np.pi  # angle in plane xz
    
    # Normalizzazione degli angoli per mantenerli entro i limiti corretti
    if angle_no_depth > 90 or angle_with_depth > 90:
        angle_no_depth -= 180
        angle_with_depth -= 180
    
    if angle_no_depth < -90 or angle_with_depth < -90:
        angle_no_depth += 180
        angle_with_depth += 180

    return angle_no_depth, angle_with_depth

def joint_rotation(joints, angle, center, type):
    if type == "no_depth":
        # Costruzione della matrice di rotazione 2D
        M = cv2.getRotationMatrix2D((center[0], center[1]), angle, 1.0)
        
        # Applicazione della matrice di rotazione
        ones = np.ones((joints.shape[0], 1))  # Aggiunge una colonna di 1 per la moltiplicazione omogenea
        joints_homogeneous = np.hstack([joints, ones])
        rotated_joints = (M.dot(joints_homogeneous.T)).T
        
        # Rimuovi la colonna aggiunta
        #rotated_joints = rotated_joints[:, :2]
        
    elif type == "with_depth":
        # Rotazione 3D nel piano xz
        cos_angle = np.cos(np.radians(angle))
        sin_angle = np.sin(np.radians(angle))
        M_xz = np.array([
            [cos_angle, 0, sin_angle],
            [0, 1, 0],
            [-sin_angle, 0, cos_angle]
        ])
        rotated_joints = (M_xz.dot(joints.T)).T

    return rotated_joints

def plot_skeleton_depth(npy, img, name_file):
    position_to_plot = [0, 5, 6, 9, 10]

    img = np.asarray(img)
    for pos in position_to_plot:
        x = 255 - npy[pos, 0]
        y = npy[pos, 1]
        img = cv2.circle(img, (int(x), int(y)), radius=int(20 * npy[pos, 2]), color=(255, 0, 0), thickness=0)
    img = Image.fromarray(img)
    img.save(f'{OUT_TEST_FOLDER}/{name_file}.png')

selected_joints = np.concatenate(( [0, 5, 6, 7, 8, 9, 10], [91, 95, 96, 99, 100, 103, 104, 107, 108, 111], [112, 116, 117, 120, 121, 124, 125, 128, 129, 132]), axis=0)

shoulder_joints = [5, 6]
connected_joints = {
    5: [5, 7, 9, 91, 95, 96, 99, 100, 103, 104, 107, 108, 111],  # Giunti connessi alla spalla sinistra
    6: [6, 8, 10, 112, 116, 117, 120, 121, 124, 125, 128, 129, 132]  # Giunti connessi alla spalla destra
}

# Path
out_folder = '/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/data-prepare/geometry/rotated_val_npy'
npy_folder = '/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/data-prepare/demo/val_npy'

for root, dirs, files in os.walk(npy_folder, topdown=False):
    for name in tqdm(files):
        npy = np.load(os.path.join(npy_folder, name)).astype(np.float32)
        npy = npy[:, selected_joints, :3] 
        
        left_shoulder = npy[:, 5, :]
        right_shoulder = npy[:, 6, :]

        center_x = np.mean([left_shoulder[:, 0], right_shoulder[:, 0]], axis=0)
        center_y = np.mean([left_shoulder[:, 1], right_shoulder[:, 1]], axis=0)
        center_z = np.mean([left_shoulder[:, 2], right_shoulder[:, 2]], axis=0)

        new_npy_no_depth = npy.copy()

        for i in range(npy.shape[0]):
            # Calcola l'angolo di rotazione
            #angle_no_depth, angle_with_depth = calculate_rotation_angle(left_shoulder[i], right_shoulder[i])
            new_left_shoulder, new_right_shoulder, M = calculate_rotation_angle(left_shoulder[i], right_shoulder[i])
            exit()
            npy_homogeneous = np.hstack([npy[i, :, :2], np.ones((npy.shape[1], 1))])
            
            # Trasformazione
            transformed_npy = (M @ npy_homogeneous.T).T
            
            # Rimuovi la colonna di 1
            new_npy_no_depth[i, :, :2] = transformed_npy[:, :2]
            
            
            # Ruota i giunti senza profondità
            #new_npy_no_depth[i, : , :2] = joint_rotation(
            #    npy[i, : , :2], 
            #    angle_no_depth, 
            #    [int(center_x[i]), int(center_y[i])], 
            #    type="no_depth"
            #)

             # Ruota i giunti collegati alle spalle
            #for shoulder, joints in connected_joints.items():
            #    for joint in joints:
            #        #new_npy_no_depth[i, joint, :2] = joint_rotation(
            #        #    npy[i, joint, :2], 
            #        #    angle_no_depth, 
            #        #    [int(center_x[i]), int(center_y[i])], 
            #        #    type="no_depth"
            #        #)
            #        new_npy_no_depth[i,joint,:2] = 

        np.save(os.path.join(out_folder, name[:-4] + '.npy'), new_npy_no_depth)
