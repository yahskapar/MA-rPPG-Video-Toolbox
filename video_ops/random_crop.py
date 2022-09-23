import os
from argparse import ArgumentParser
from xmlrpc.client import boolean
import mat73
import numpy as np
import imageio
from tqdm import tqdm
from PIL import Image, ImageOps
import hdf5storage
import warnings
warnings.filterwarnings("ignore")

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
    parser.add_argument("--no_upscale", default=False, action='store_true', help="don't upscale the video") 
    parser.add_argument("--center", default=False,action='store_true', help="crop to the center") 
    parser.add_argument("--leave_black", default=False,action='store_true',help="resize to new sizes but leave black instead of upscale") 
    opt = parser.parse_args()
    idx = 0
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    for video_path in tqdm(os.listdir(opt.video_path)):
        start_row = np.random.randint(240-opt.new_width)
        start_col = np.random.randint(240-opt.new_length)
        mat, video = read_video(os.path.join(opt.video_path,video_path))
        video = video / video.max() #normalizes img_grey in range 0 - 255
        video = (255 * video).astype(np.uint8)
        result = []
        for frame in video:
            if opt.center:
                crop_img = frame[120-opt.new_length//2:120+opt.new_length//2,120-opt.new_width//2:120+opt.new_width//2,:]
            else:
                crop_img = frame[start_row:start_row+opt.new_length,start_col:start_col+opt.new_width,:]
            img = Image.fromarray(crop_img.astype('uint8'), 'RGB')
            if not opt.no_upscale: 
                img = img.resize((240,240))
            if opt.leave_black:
                img = ImageOps.expand(img, border = (240-img.size[0])//2, fill = 0)
                img = img.resize((240,240))
            result.append(np.array(img,dtype=np.uint8))
        #save_video(mat,video_path.split('.')[0],result,opt.save_path)
        result_file = opt.save_path+"/result_"+str(idx)+".mp4"  
        imageio.mimsave(result_file, result, fps = 30)
        idx+=1

if __name__ == '__main__':
    main()
    