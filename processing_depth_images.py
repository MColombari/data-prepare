import os
from tqdm import tqdm
import cv2
import numpy as np
from scipy.spatial import KDTree

OG_VIDEO_PATH = "/work/cvcs2024/SLR_sentiment_enhanced/datasets/custom/raw_video"
DEPTH_VIDEO_PATH = "/work/cvcs2024/SLR_sentiment_enhanced/datasets/custom/depth_video_local"
OUT_PATH = "/work/cvcs2024/SLR_sentiment_enhanced/datasets/custom/depth_video_local_optimized"

# The depth is encoded with cv2.COLORMAP_INFERNO color map, we want to have a reverse.
def inverse_color_map(colormap):
    inverse_colormap = {}
    for i in tqdm(range(256)):
        color = cv2.applyColorMap(np.array([[i]], dtype=np.uint8), colormap)[0, 0]
        inverse_colormap[tuple(color)] = i
    return inverse_colormap

def build_kdtree(inv_map):
    # Builds a k-d tree for fast nearest color lookup.
    colors = np.array(list(inv_map.keys()))
    return KDTree(colors), colors


# Create inverse color map.
inv_map = inverse_color_map(cv2.COLORMAP_INFERNO)
kdtree, colors = build_kdtree(inv_map)


def pixel_level_change_color(px):
    color = tuple(px)
    # Find the closest match in the inverse colormap
    if color in inv_map.keys():
        px[0] = inv_map[color]
    else:
        dist, idx = kdtree.query(color)
        closest_color = tuple(colors[idx])
        px[0] = inv_map[closest_color]


def change_color(img, inv_map, kdtree, colors):
    depth_blue_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            color = tuple(img[y, x])

            # Find the closest match in the inverse colormapù
            # print(color)
            # print(inv_map.keys())
            # assert color in inv_map.keys()
            if color in inv_map.keys():
                depth_blue_image[y, x, 0] = inv_map[color]
            else:
                dist, idx = kdtree.query(color)
                closest_color = tuple(colors[idx])
                depth_blue_image[y, x, 0] = inv_map[closest_color]
                inv_map[tuple(color)] = inv_map[closest_color]

    
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


# Resize and change color image
for root, dirs, files in os.walk(DEPTH_VIDEO_PATH, topdown=False):
    for name in tqdm(files):
        if 'depth' in name:
            if not 'signer0_orientationLeft_sample0' in name:
                continue # Skip this now.

            og_width, og_height = og_size_dict[name[:-16]]
            frames = []
            cap = cv2.VideoCapture(os.path.join(root, name))
            while True:
                ret, frame = cap.read()
                if ret:
                    # print(frame.shape)
                    cimage = frame[:, -og_width:]
                    assert cimage.shape[0] == og_height and cimage.shape[1] == og_width

                    # np.apply_along_axis(pixel_level_change_color, 2, cimage)
                    fimage = change_color(cimage, inv_map, kdtree, colors)
                    frames.append(fimage)
                else:
                    break

            # write to MP4 file
            vidwriter = cv2.VideoWriter(OUT_PATH + "/" + name[:-16] + "_depth_opt.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (og_width, og_height))
            for frame in frames:
                vidwriter.write(frame)
            vidwriter.release()