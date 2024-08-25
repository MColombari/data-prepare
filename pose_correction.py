import cv2
import numpy as np
import math
import os
from wholepose.utils import plot_31_pose
            

# Selection shoulders' joints
# Calculate rotation angle
# Rotate joints
# Centers allignment between shoulders and frame

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


def rotate_image_and_joints(joints, angle, center, type):

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


selected_joints = np.concatenate(([0,1,2,3,4,5,6,7,8,9,10], 
                    [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0) 
shoulder_joints=[5,6]

# Path
image_path = '/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/data-prepare/train_frames/WLASL/signer4_sample334/0011.jpg'
#output_path = '/homes/omoussadek/CV/SLR_Sentiment_Enhanced/data-prepare/signer4_sample334/0011_correct_pose.jpg'
npy_path='/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/data-prepare/demo/train_npy/signer4_sample334_color.mp4.npy'



# Load npy data
npy = np.load(npy_path).astype(np.float32)
print(npy[1,5,:])
print(npy[1,6,:])
npy = npy[:, selected_joints, :2]

print(npy[1,shoulder_joints,:])

# Image loading
image = cv2.imread(image_path)

# Selection shoulders' joints
left_shoulder = npy[:, 5, :]
right_shoulder = npy[:, 6, :]

center_x = np.mean([left_shoulder[:,0],right_shoulder[:,0]], axis=0)
center_y = np.mean([left_shoulder[:,1],right_shoulder[:,1]], axis=0)
center_z = np.mean([left_shoulder[:,2],right_shoulder[:,2]], axis=0)
print(f'center_z: {center_z[1]}')

#print(f'left_sh: {left_shoulder[1]}')
#print(f'right_sh: {right_shoulder[1]}')
#print(f'center_x: {center_x[1]}')
#print(f'center_y: {center_y[1]}')

#print(center[0])


new_npy_no_deepth=npy.copy()
new_npy_with_deepth=npy.copy()

for i in range(npy.shape[0]):
    # Calculate rotation angle
    angle_no_deepth, angle_with_deepth = calculate_rotation_angle(left_shoulder[i], right_shoulder[i])
    new_npy_no_deepth[i,:,:2] = rotate_image_and_joints(npy[i,:,:2], angle_no_deepth, [int(center_x[i]), int(center_y[i])], type="no_deepth")
    if i == 1:
        print(f'angle_no_deepth:{angle_no_deepth}, angle_with_deepth:{angle_with_deepth}')
        print(new_npy_with_deepth[1,shoulder_joints,:])
        exit()
    #new_npy_with_deepth[i,:,:] = rotate_image_and_joints(npy[i,:,:], angle_with_deepth, [int(center_x[i]), int(center_y[i]), int(center_z[i])], type="with_deepth")



#print(new_npy_no_deepth[1,shoulder_joints,:])



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