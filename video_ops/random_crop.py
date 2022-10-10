import os
from argparse import ArgumentParser
from xmlrpc.client import boolean
import mat73
import numpy as np
import imageio
from tqdm import tqdm
import cv2
import hdf5storage
import warnings
warnings.filterwarnings("ignore")

from torchvideotransforms import video_transforms, volume_transforms
from skimage import img_as_ubyte

def read_video(video_file):
    """Reads a video file, returns frames(T,H,W,3) """
    mat = mat73.loadmat(video_file)
    frames = mat['Xsub']  # load raw frames
    return mat,np.asarray(frames)

def save_video(mat, file_name, new_xsub, save_path):
    """Reads a video file, returns frames(T,H,W,3) """
    mat['Xsub'] = new_xsub # save raw frames
    filename =  file_name + '_random_crop.mat'
    hdf5storage.savemat(os.path.join(save_path, filename), mat, format='7.3')
    return


def main():
    parser = ArgumentParser()
    parser.add_argument("--video_path", default='scamps_videos_example', help="path to videos")
    parser.add_argument("--save_path", default='result_random_crop', help="path to save videos")
    parser.add_argument("--new_length", default=160, help="new length")
    parser.add_argument("--new_width", default=160, help="new width")
    parser.add_argument("--upscale", default=False, action='store_true', help="upscale the video") 
    parser.add_argument("--center", default=False,action='store_true', help="crop to the center") 
    opt = parser.parse_args()
    idx = 0
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    for video_path in tqdm(os.listdir(opt.video_path)):
        mat, video = read_video(os.path.join(opt.video_path,video_path))
        if opt.center:
            video_transform_list = [video_transforms.CenterCrop((opt.new_length,opt.new_width))]
        else:
            video_transform_list = [video_transforms.RandomCrop((opt.new_length,opt.new_width))]
        transforms = video_transforms.Compose(video_transform_list)
        result = transforms(video)

        if opt.upscale:
            t = []
            for frame in result:
                t.append(cv2.resize(frame,(240,240),interpolation = cv2.INTER_LINEAR))
            result = t
        save_video(mat,video_path.split('.')[0],result,opt.save_path)
        #result_file = opt.save_path+"/result_"+str(idx)+".mp4"  
        #imageio.mimsave(result_file, [img_as_ubyte(f) for f in result], fps = 30)
        idx+=1

if __name__ == '__main__':
    main()

#python random_crop --upscale
    