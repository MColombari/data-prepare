import os
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

npy_folder_input = '/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/data-prepare/demo/custom_dataset_npy'
folder = '/work/cvcs2024/SLR_sentiment_enhanced/datasets/custom/depth_video_local_optimized' #quello con i video
npy_folder_output = '/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/data-prepare/demo/custom_dataset_depth_npy'

selected_joints = np.concatenate(([0,5,6,7,8,9,10], 
                    [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0)



#per ogni video del folder 
    #considera il npy corrispondente
    #salva il numero dei frame del video 
    #for i in frames
        #prendi il frame del video corrispondente
        
        #trova il valore di blu in posizione x, y di joint--> z
        #salva[i,joint,(x,y,z)]
        
    #salva questo npy in una cartella nel condiviso

#/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/data-prepare/demo/custom_dataset_npy/signer0_orientationLeft_sample0.mp4.npy --> esempio di file npy
#/work/cvcs2024/SLR_sentiment_enhanced/datasets/custom/depth_video_local_optimized/signer0_orientationLeft_sample0_depth_opt.mp4 --> esempio percorso video

def countFrame (path):
    cap = cv2.VideoCapture(path)
    counter = 0
    while True:
        ret, _ = cap.read()
        if ret:
            counter += 1
            # print(frame.shape)
        else:
            break
    cap.release()
    return counter
            
            
            
for root, dirs, files in os.walk(folder, topdown=False):
    for name in tqdm(files):
        if '_depth_opt' in name:
            video_path = os.path.join(root, name)
            npy_name = name.replace('_depth_opt', '')
            
            if not os.path.exists(os.path.join(npy_folder_input, npy_name + '.npy')):
                continue
            
            if os.path.exists(os.path.join(npy_folder_output, npy_name + '.npy')):
                continue
            
            frame_video = countFrame(video_path)
            #print(f"Frame count: {frame_count}") 

            print(f"Trying to open video: {video_path}")  # (debug)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                #print(f"Failed to open video: {video_path}")
                continue  
            
            
            
            npy_input_path=os.path.join(npy_folder_input, npy_name + '.npy')
            npy = np.load(os.path.join(npy_folder_input, npy_name + '.npy')).astype(np.float32)
            print(f"Open npy file: {npy_input_path}")
            npy = npy[:, selected_joints, :3]
            frame_npy = npy.shape[0]
            
            depth_frames = np.zeros((frame_npy, len(selected_joints), 3), dtype=np.float32)

            sampling_rate = frame_video // frame_npy

            npy_out= np.zeros((frame_npy, len(selected_joints), 4), dtype=np.float32) #x,y,z,score
            
            for i in range(frame_npy):
                frame_number = i * sampling_rate
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if not ret:
                    #print(f"Failed to read frame {i} from video: {video_path}")
                    break
                
                
                for j, joint in enumerate(selected_joints):
                    x, y = int(npy[i, j, 0]), int(npy[i, j, 1])  
                    npy_out[i, j, :2] = npy[i, j, :2]
                    npy_out[i ,j ,3] = npy[i, j, 2]  # Profondità z dal file npy
                    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:  # Verifica i limiti
                        mulx = frame.shape[1] / 256
                        muly = frame.shape[0] / 256
                        xframe = int(x*mulx)
                        yframe = int(y*muly)
                        # Leggi il valore del canale Blu dal frame
                        depth_value = frame[yframe, xframe, 0]  # Canale Blu in BGR (?) 
                        npy_out[i,j,2]=depth_value
                        #if depth_value != 0:
                        #    print(joint,j)
                        #    print(npy[i,j,:]) #-> per xy
                        #    print(depth_value)
                        #    exit()
            
            cap.release()
            npy_output_path = os.path.join(npy_folder_output, npy_name + '.npy')
            np.save(os.path.join(npy_folder_output, npy_name + '.npy'), npy_out)
            print(f"New npy file created: {npy_input_path}")
            