import os
from argparse import ArgumentParser
import mat73
import numpy as np
import imageio
from scipy.ndimage.interpolation import rotate
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def read_video(video_file):
    """Reads a video file, returns frames(T,H,W,3) """
    mat = mat73.loadmat(video_file)
    frames = mat['Xsub']  # load raw frames
    t = np.asarray(frames)
    t = t / t.max() #normalizes img_grey in range 0 - 255
    t = (255 * t).astype(np.uint8)
    return t


def main():
    parser = ArgumentParser()
    parser.add_argument("--video_path", default='scamps_videos_example', help="path to videos")
    parser.add_argument("--rotation_degree", default=10,type=int, help="rotation degree")
    parser.add_argument("--save_path", default='result_rotation', help="path to save videos")
    parser.add_argument("--mode", default='nearest', help="to fill the blank space of rotation. use constant to leave black")
    opt = parser.parse_args()
    idx = 0
    mode = opt.mode
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    for video in tqdm(os.listdir(opt.video_path)):
        video = read_video(os.path.join(opt.video_path,video))
        result = []
        for frame in video:
            img = rotate(frame, angle=opt.rotation_degree, reshape=False, mode=mode)
            result.append(img)
        result_file = os.path.join(opt.save_path,"result_"+str(idx)+".mp4") 
        imageio.mimsave(result_file, result, fps = 30)
        idx+=1

if __name__ == '__main__':
    main()
    