import cv2
import numpy as np
import os

def crop(image, center, radius, size=512):
    scale = 1.3
    radius_crop = (radius * scale).astype(np.int32)
    center_crop = (center).astype(np.int32)

    rect = (max(0,(center_crop-radius_crop)[0]), max(0,(center_crop-radius_crop)[1]), 
                 min(size,(center_crop+radius_crop)[0]), min(size,(center_crop+radius_crop)[1]))

    image = image[rect[1]:rect[3],rect[0]:rect[2],:]

    if image.shape[0] < image.shape[1]:
        top = abs(image.shape[0] - image.shape[1]) // 2
        bottom = abs(image.shape[0] - image.shape[1]) - top
        image = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT,value=(0,0,0))
    elif image.shape[0] > image.shape[1]:
        left = abs(image.shape[0] - image.shape[1]) // 2
        right = abs(image.shape[0] - image.shape[1]) - left
        image = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT,value=(0,0,0))
    return image

selected_joints = np.concatenate(([0,1,2,3,4,5,6,7,8,9,10], 
                    [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0) 
folder = '/work/cvcs2024/SLR_sentiment_enhanced/datasets/WLASL/WLASL/start_kit/data/val' # 'train', 'test'
npy_folder = '/work/cvcs2024/SLR_sentiment_enhanced/tmp2' #'val_npy/npy3' # 'train_npy/npy3', 'test_npy/npy3'
out_folder = 'prova_frames' #'/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/data-prepare/val_frames/WLASL' # 'train_frames' 'test_frames'



def plot_31_pose(image, center_p, keypoints_joints, scale=((1.0,1.0))):
    for n in range(keypoints_joints.shape[0]):
        cor_x, cor_y = int(keypoints_joints[n, 0] * scale[0]), int(keypoints_joints[n, 1] * scale[1])

        frame_height, frame_width = image.shape[:2]
        assert cor_x < frame_height
        assert cor_y < frame_width

        image = cv2.circle(image, (cor_x, cor_y), radius=2, color=(255,0,0), thickness=-1)
    # draw center   
    image = cv2.circle(image, (int(center_p[0]), int(center_p[1])), radius=2, color=(0, 0, 255), thickness=-1)

    return image


for root, dirs, files in os.walk(folder, topdown=False):
    for name in files:
        if 'color' in name:
            print(os.path.join(root, name))
            if 'signer11' not in name:
                continue
            if  not os.path.exists(os.path.join(npy_folder, name + '.npy')):
                continue
            cap = cv2.VideoCapture(os.path.join(root, name))
            npy = np.load(os.path.join(npy_folder, name + '.npy')).astype(np.float32)
            npy = npy[:, selected_joints, :2]
            print(npy.shape)
            # npy[:, :, 0] = 512 - npy[:, :, 0]
            xy_max = npy.max(axis=1, keepdims=False).max(axis=0, keepdims=False)
            xy_min = npy.min(axis=1, keepdims=False).min(axis=0, keepdims=False)
            assert xy_max.shape == (2,)
            xy_center = (xy_max + xy_min) / 2 # - 20 why?!?!?!?!?
            xy_radius = (xy_max - xy_center).max(axis=0)
            index = 0
            while True:
                ret, frame = cap.read()
                if ret:
                    height, width, channels = frame.shape

                    print(f'image size {height}, {width}')

                    frame = cv2.resize(frame, (256,256))

                    img = plot_31_pose(frame, xy_center, npy[index])
                    cv2.imwrite(os.path.join(out_folder, name[:-10], '{:04d}_non_crop.jpg'.format(index+1)), img)
                    print(os.path.join(out_folder, name[:-10], '{:04d}_non_crop.jpg'.format(index+1)))

                    print(f'xy_max: {xy_max}, xy_min:{xy_min}')
                    print(f'center: {xy_center}')
                    print(frame.shape)
                    
                    
                    image = crop(frame, xy_center, xy_radius)
                else:
                    break
                index = index + 1
                image = cv2.resize(image, (256,256))
                if not os.path.exists(os.path.join(out_folder, name[:-10])):
                    os.makedirs(os.path.join(out_folder, name[:-10]))
                cv2.imwrite(os.path.join(out_folder, name[:-10], '{:04d}.jpg'.format(index)), image)
                print(os.path.join(out_folder, name[:-10], '{:04d}.jpg'.format(index)))

