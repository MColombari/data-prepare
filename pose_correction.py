import cv2
import numpy as np
import math
import os
from wholepose.utils import plot_31_pose

from PIL import Image
            

OUT_TEST_FOLDER = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/data-prepare/test_folder"

# Selection shoulders' joints
# Calculate rotation angle
# Rotate image and joints
# Centers allignment between shoulders and frame

def calculate_rotation_angle(joint1, joint2):
    delta_y = joint2[1] - joint1[1]
    delta_x = joint2[0] - joint1[0]
    
    angle = np.arctan2(delta_y, delta_x)
    return angle

def rotate_image_and_joints(joints, angle, center):
    # Rotate image
    #h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
    #rotated_image = cv2.warpAffine(image, M, (w, h))

    # Rotate joints
    ones = np.ones(shape=(joints.shape[0], 1))
    joints_homogeneous = np.hstack([joints, ones])
    rotated_joints = M.dot(joints_homogeneous.T).T
    
    
    return rotated_joints

def translate_image_and_joints(image, joints, translation):
    # Translate image
    h, w = image.shape[:2]
    M = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
    translated_image = cv2.warpAffine(image, M, (w, h))

    # Translate joints
    translated_joints = joints + translation
    
    return translated_image, translated_joints


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


selected_joints = np.concatenate(([0,1,2,3,4,5,6,7,8,9,10], 
                    [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0) 
shoulder_joints=[5,6]

# Path
image_path = '/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/data-prepare/train_frames/WLASL/signer4_sample334/0003.jpg'
#output_path = '/homes/omoussadek/CV/SLR_Sentiment_Enhanced/data-prepare/signer4_sample334/0011_correct_pose.jpg'
npy_path='/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/data-prepare/demo/train_npy/signer4_sample334_color.mp4.npy'
name_file = image_path[-8:-4]
print(name_file)

# Image loading
image = cv2.imread(image_path)

# Load npy data
npy = np.load(npy_path).astype(np.float32)
plot_skeleton_depth(npy[10, :, :], image, name_file)
npy = npy[:, selected_joints, :2]

print(npy[1,shoulder_joints,:])

# Selection shoulders' joints
left_shoulder = npy[:, 5, :]
right_shoulder = npy[:, 6, :]
center_x = np.mean([left_shoulder[:,0],right_shoulder[:,0]], axis=0)
center_y = np.mean([left_shoulder[:,1],right_shoulder[:,1]], axis=0)


#print(f'left_sh: {left_shoulder[1]}')
#print(f'right_sh: {right_shoulder[1]}')
#print(f'center_x: {center_x[1]}')
#print(f'center_y: {center_y[1]}')

#print(center[0])

new_npy=np.zeros(npy.shape)

for i in range(npy.shape[0]):
    # Calculate rotation angle
    angle = calculate_rotation_angle(left_shoulder[i], right_shoulder[i])
    if i == 1:
        print(f'angle:{angle}')
    new_npy[i,:,:] = rotate_image_and_joints(npy[i,:,:], angle, [int(center_x[i]), int(center_y[i])])


print(new_npy[1,shoulder_joints,:])


exit()
# Centers allignment between shoulders and frame
frame_size = 256
translation = np.array([frame_size / 2, frame_size / 2]) - center
translated_image, translated_joints = translate_image_and_joints( rotated_joints, translation)

# Resize traslated image to the 256x256 frame
resized_image = cv2.resize(translated_image, (frame_size, frame_size))

# Crea la directory di output se non esiste
#os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Salva l'immagine raddrizzata e centrata
cv2.imwrite(output_path, resized_image)

# Visualizza i joints sull'immagine ridimensionata
resized_image_with_pose = plot_31_pose(resized_image, center, translated_joints)

# Salva l'immagine con i joints disegnati
output_pose_path = output_path.replace(".jpg", "_pose.jpg")
cv2.imwrite(output_pose_path, resized_image_with_pose)

#print(f"Immagine raddrizzata e centrata salvata in: {output_path}")
#print(f"Immagine con joints disegnati salvata in: {output_pose_path}")
