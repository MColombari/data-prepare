import os
from tqdm import tqdm
import cv2
import numpy as np
from scipy.spatial import KDTree

OG_VIDEO_PATH = "/work/cvcs2024/SLR_sentiment_enhanced/datasets/custom/raw_video"
DEPTH_VIDEO_PATH = "/work/cvcs2024/SLR_sentiment_enhanced/datasets/custom/depth_video_local"

# The depth is encoded with cv2.COLORMAP_INFERNO color map, we want to have a reverse.
def inverse_color_map(colormap):
    inverse_colormap = {}
    for i in tqdm(range(255)):
        color = cv2.applyColorMap(np.array([[i]], dtype=np.uint8), colormap)[0, 0]
        inverse_colormap[tuple(color)] = i
    return inverse_colormap

def build_kdtree(inv_map):
    # Builds a k-d tree for fast nearest color lookup.
    colors = np.array(list(inv_map.keys()))
    return KDTree(colors), colors

def change_color(img, inv_map, kdtree, colors):
    depth_blue_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for y in tqdm(range(img.shape[0])):
        for x in range(img.shape[1]):
            color = tuple(img[y, x])

            # Find the closest match in the inverse colormap
            if color in inv_map.keys():
                depth_blue_image[y, x, 0] = inv_map[color]
            else:
                dist, idx = kdtree.query(color)
                closest_color = tuple(colors[idx])
                depth_blue_image[y, x, 0] = inv_map[closest_color]
    
    return depth_blue_image


# Read original size.
og_size_dict = {}
for root, dirs, files in os.walk(OG_VIDEO_PATH, topdown=False):
    for name in tqdm(files):
        if 'signer' in name:
            cap = cv2.VideoCapture(os.path.join(root, name))
            if cap.isOpened(): 
                width  = int(cap.get(3))  # float `width`
                height = int(cap.get(4))  # float `height`
                # print(name[:-4])
                og_size_dict[name[:-4]] = (width, height)


# Create inverse color map.
inv_map = inverse_color_map(cv2.COLORMAP_INFERNO)
kdtree, colors = build_kdtree(inv_map)

# Resize and change color image
for root, dirs, files in os.walk(DEPTH_VIDEO_PATH, topdown=False):
    for name in files:
        if 'depth' in name:
            if 'signer0_orientationLeft_sample0' in name:
                continue # Skip this now.

            og_width, og_height = og_size_dict[name[:-16]]
            cap = cv2.VideoCapture(os.path.join(root, name))
            while True:
                ret, frame = cap.read()
                if ret:
                    # print(frame.shape)
                    cimage = frame[:, -og_width:]
                    assert cimage.shape[0] == og_height and cimage.shape[1] == og_width

                    fimage = change_color(cimage, inv_map, kdtree, colors)
                    cv2.imwrite("Cropped Image.jpg", fimage)
                    exit()
                else:
                    break