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

        new_npy_no_deepth = npy.copy()

        for i in range(npy.shape[0]):
            angle_no_deepth, angle_with_deepth = calculate_rotation_angle(left_shoulder[i], right_shoulder[i])

            # Apply rotation to shoulders
            center = (int(center_x[i]), int(center_y[i]))
            rotated_shoulders = rotate_joints_2d(npy[i, shoulder_joints, :2], angle_no_deepth, center)
            new_npy_no_deepth[i, shoulder_joints, :2] = rotated_shoulders

            # Compute the vertical delta for translation
            original_shoulder_y = np.mean(npy[i, shoulder_joints, 1])
            rotated_shoulder_y = np.mean(rotated_shoulders[:, 1])
            delta_y = rotated_shoulder_y - original_shoulder_y

            # Determine the proximity of other joints to the right shoulder
            distances_to_right_shoulder = [calculate_distance(npy[i, j, :], right_shoulder[i]) for j in range(npy.shape[1])]
            distances_to_left_shoulder = [calculate_distance(npy[i, j, :], left_shoulder[i]) for j in range(npy.shape[1])]
            proximity_indices = [idx for idx in range(npy.shape[1]) if distances_to_right_shoulder[idx] < distances_to_left_shoulder[idx]]

            # Apply translation to joints closer to the right shoulder
            for joint_idx in proximity_indices:
                new_npy_no_deepth[i, joint_idx, :] = apply_translation(npy[i, joint_idx, :], delta_y, axis=1)

        np.save(os.path.join(out_folder, name[:-4] + '.npy'), new_npy_no_deepth)
