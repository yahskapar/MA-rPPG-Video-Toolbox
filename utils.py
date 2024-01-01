import os, glob, yaml, shutil
import numpy as np
import torch
import cv2
import mat73
import hdf5storage

from sync_batchnorm import DataParallelWithCallback
from modules.generator import OcclusionAwareGenerator, OcclusionAwareSPADEGenerator
from modules.keypoint_detector import KPDetector, HEEstimator

def copy_folder(src_folder, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    for src_dir, dirs, files in os.walk(src_folder):
        dst_dir = src_dir.replace(src_folder, dst_folder, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file_ in files:
            if file_.endswith('.avi') or file_.endswith('.png') or file_.endswith('.mat'):
                continue
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                # os.remove(dst_file)
                continue
            shutil.copy2(src_file, dst_dir)

def load_checkpoints(config_path, checkpoint_path, gen, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if gen == 'original':
        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
    elif gen == 'spade':
        generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                                 **config['model_params']['common_params'])

    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()

    he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                               **config['model_params']['common_params'])
    if not cpu:
        he_estimator.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    # TODO: Investigate any mismatches, not clear why this is suddenly happening
    # and thus requires strict=False option
    generator.load_state_dict(checkpoint['generator'], strict=False)
    kp_detector.load_state_dict(checkpoint['kp_detector'], strict=False)
    he_estimator.load_state_dict(checkpoint['he_estimator'], strict=False)
    
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)
        he_estimator = DataParallelWithCallback(he_estimator)

    generator.eval()
    kp_detector.eval()
    he_estimator.eval()
    
    return generator, kp_detector, he_estimator

def read_ubfc_video(video_file):
        """Reads a video file, returns frames(T,H,W,3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        print(np.shape(frames))
        return np.asarray(frames)

def read_pure_video(video_file):
    """Reads a video file, returns frames(T, H, W, 3) """
    frames = list()
    all_png = sorted(glob.glob(video_file + '*.png'))
    for png_path in all_png:
        img = cv2.imread(png_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    return np.asarray(frames)

# Read_video to convert video file to numpy array
def read_scamps_video(video_file):
    """Reads a video file, returns frames(T,H,W,3) """
    mat = mat73.loadmat(video_file)
    frames = mat['Xsub']  # load raw frames
    return np.asarray(frames)

# Save_video to write a new Xsub to corresponding .mat file
def save_scamps_video(video_file_path, video_file, driving_video_file, new_xsub, save_path):
    """Reads a video file, returns frames(T,H,W,3) """
    mat = mat73.loadmat(os.path.join(video_file_path, video_file))
    mat['Xsub'] = new_xsub # save raw frames
    source_video_name = os.path.splitext(video_file)[0]
    driving_video_name = os.path.splitext(driving_video_file)[0]
    filename = source_video_name + '_' + driving_video_name + '.mat'
    hdf5storage.savemat(os.path.join(save_path, filename), mat, format='7.3')
    return
