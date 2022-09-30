import os
from argparse import ArgumentParser
import mat73
import numpy as np
import imageio
from tqdm import tqdm
import hdf5storage
from skimage import img_as_ubyte
import warnings
import cv2
warnings.filterwarnings("ignore")


def read_video(video_file):
    """Reads a video file, returns frames(T,H,W,3) """
    mat = mat73.loadmat(video_file)
    frames = mat['Xsub']  # load raw frames
    return mat,np.asarray(frames)

def save_video(mat, file_name, new_xsub, save_path):
    """Reads a video file, returns frames(T,H,W,3) """
    mat['Xsub'] = new_xsub # save raw frames
    filename =  file_name + '_rotation.mat'
    hdf5storage.savemat(os.path.join(save_path, filename), mat, format='7.3')
    return


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def main():
    parser = ArgumentParser()
    parser.add_argument("--video_path", default='scamps_videos_example', help="path to videos")
    parser.add_argument("--rotation_degree", default=10,type=int, help="rotation degree")
    parser.add_argument("--save_path", default='result_rotation', help="path to save videos")
    opt = parser.parse_args()
    idx = 0
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    for video_path in tqdm(os.listdir(opt.video_path)):
        mat, video = read_video(os.path.join(opt.video_path,video_path))
        result = []
        for frame in video:
            t = rotate_image(frame,opt.rotation_degree)
            result.append(t)

        save_video(mat,video_path.split('.')[0],result,opt.save_path)
        #result_file = opt.save_path+"/result_"+str(idx)+".mp4"  
        #imageio.mimsave(result_file, [img_as_ubyte(f) for f in result], fps = 30)
        idx+=1

if __name__ == '__main__':
    main()
    