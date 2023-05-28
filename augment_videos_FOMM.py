import matplotlib
matplotlib.use('Agg')
import os, sys, glob
import time
import shutil
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import gc

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
import torch.nn.functional as F
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp

import ffmpeg
from os.path import splitext
from shutil import copyfileobj
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import mat73
import scipy.io
from scipy.sparse import spdiags
from scipy.signal import butter
import hdf5storage

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import cv2
from vidgear.gears import VideoGear

from torch.multiprocessing import Pool, Process, Value, Array, Manager, set_start_method


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.full_load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector

def make_animation(gpu, source_image, driving_video, frame_num, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        if not cpu:
            driving = driving.cuda()
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        driving_frame = driving[:, :, frame_num]
        if not cpu:
            driving_frame = driving_frame.cuda()
        kp_driving = kp_detector(driving_frame)
        kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
        out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

        predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    # Clean-up
    del source
    del driving
    del driving_frame
    return predictions

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
def save_video(video_file_path, video_file, driving_video_file, new_xsub, save_path):
    """Reads a video file, returns frames(T,H,W,3) """
    mat = mat73.loadmat(os.path.join(video_file_path, video_file))
    # new_xsub = np.array(new_xsub, dtype=np.float64)
    # new_xsub = new_xsub / 255
    # new_xsub.tolist()
    mat['Xsub'] = new_xsub # save raw frames
    source_video_name = os.path.splitext(video_file)[0]
    driving_video_name = os.path.splitext(driving_video_file)[0]
    filename = source_video_name + '_' + driving_video_name + '.mat'
    hdf5storage.savemat(os.path.join(save_path, filename), mat, format='7.3')
    return

def make_video(dataset, gpu, opt, source_video, driving_video,generator,kp_detector,source_directory, source_filename, driving_filename, augmented_path):
    final_preds = []
    # Set GPU
    torch.cuda.set_device(gpu)

    # The progress bar will effectively be broken when multi-processing is used
    # A fix will be implemented in a future update to this toolbox
    frames_pbar = tqdm(list(range(min(np.shape(source_video)[0], np.shape(driving_video)[0]))))
    
    # for frames in tqdm(range(min(np.shape(source_video)[0], np.shape(driving_video)[0]))):
    for frames in range(min(np.shape(source_video)[0], np.shape(driving_video)[0])):
        source_image = resize(source_video[frames], (256, 256))[..., :3]
        #print(f'estimate jacobian: {estimate_jacobian}')
        predictions = make_animation(gpu, source_image, driving_video, frames, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
        final_preds.append(predictions)
        frames_pbar.update(1)
    frames_pbar.close()
    np_preds = np.squeeze(np.asarray(final_preds))

    # ### Debugging
    # final_preds_save = [resize(frame, (240, 240))[..., :3] for frame in np_preds]
    # source_video_name = os.path.splitext(source_filename)[0]
    # driving_video_name = os.path.splitext(driving_filename)[0]
    # filename = source_video_name + '_' + driving_video_name + '.mp4'
    # print('saved: ', os.path.join(augmented_path, filename))
    # imageio.mimsave(os.path.join(augmented_path, filename), [img_as_ubyte(frame) for frame in final_preds_save], fps=30)
    # ###

    # exit()

    if dataset == 'SCAMPS':
        final_preds = [resize(frame, (240, 240))[..., :3] for frame in np_preds]
        save_video(source_directory, source_filename, driving_filename, final_preds, augmented_path)
    elif dataset == 'UBFC-rPPG':
        final_preds = [resize(frame, (480, 640))[..., :3] for frame in np_preds]
        source_video_name = os.path.splitext(source_filename)[0]
        driving_video_name = os.path.splitext(driving_filename)[0]
        filename = source_video_name + '_' + driving_video_name + '.npy'
        np.save(os.path.join(augmented_path, source_video_name, filename), final_preds)
    elif dataset == 'UBFC-PHYS':
        final_preds = [resize(frame, (1024, 1024))[..., :3] for frame in np_preds]
        source_video_name = os.path.splitext(source_filename)[0]
        driving_video_name = os.path.splitext(driving_filename)[0]
        filename = source_video_name + '_' + driving_video_name + '.npy'
        np.save(os.path.join(augmented_path, source_video_name, filename), final_preds)
    elif dataset == 'PURE':
        final_preds = [resize(frame, (480, 640))[..., :3] for frame in np_preds]
        source_video_name = os.path.splitext(source_filename)[0]
        driving_video_name = os.path.splitext(driving_filename)[0]
        filename = source_video_name + '_' + driving_video_name + '.npy'
        np.save(os.path.join(augmented_path, source_video_name, source_video_name, filename), final_preds)
    
    # Clean-up
    gc.collect()
    torch.cuda.empty_cache()
    return

# Resize back to original size
def resize_to_original(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def face_detection(frame, use_larger_box=False, larger_box_coef=1.0):
    """Face detection on a single frame.

    Args:
        frame(np.array): a single frame.
        use_larger_box(bool): whether to use a larger bounding box on face detection.
        larger_box_coef(float): Coef. of larger box.
    Returns:
        face_box_coor(List[int]): coordinates of face bouding box.
    """
    detector = cv2.CascadeClassifier('./utils/haarcascade_frontalface_default.xml')
    face_zone = detector.detectMultiScale(frame)
    if len(face_zone) < 1:
        print("ERROR: No Face Detected")
        face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
    elif len(face_zone) >= 2:
        face_box_coor = np.argmax(face_zone, axis=0)
        face_box_coor = face_zone[face_box_coor[2]]
        print("Warning: More than one faces are detected(Only cropping the biggest one.)")
    else:
        face_box_coor = face_zone[0]
    if use_larger_box:
        face_box_coor[0] = max(0, face_box_coor[0] - (larger_box_coef - 1.0) / 2 * face_box_coor[2])
        face_box_coor[1] = max(0, face_box_coor[1] - (larger_box_coef - 1.0) / 2 * face_box_coor[3])
        face_box_coor[2] = larger_box_coef * face_box_coor[2]
        face_box_coor[3] = larger_box_coef * face_box_coor[3]
    return face_box_coor

def augment_motion(dataset, gpu, source_list, driving_list, augmented_list, i, opt, running_num, source_directory, driving_directory, generator, kp_detector):
    # TODO: Add error handling throughout this function
    
    source_filename = os.fsdecode(source_list[i])

    if dataset == 'SCAMPS':
        source_video = []
        source_video = read_scamps_video(os.path.join(source_directory, source_filename))
        source_video.tolist()
        print("source: ",os.path.join(source_directory, source_filename))
    elif dataset == 'UBFC-rPPG':
        source_video = []
        print("source: ",os.path.join(source_directory, source_filename, 'vid.avi'))
        source_video = read_ubfc_video(os.path.join(source_directory, source_filename, 'vid.avi')) 
        source_video.tolist()
    elif dataset == 'UBFC-PHYS':
        source_video = []
        print("source: ",os.path.join(source_directory, source_filename, f'vid_{source_filename}_T1.avi'))
        source_video = read_ubfc_video(os.path.join(source_directory, source_filename, f'vid_{source_filename}_T1.avi'))
        print("I got out of the read function!")
        # source_video.tolist()
    elif dataset == 'PURE':
        source_video = []
        print("source: ",os.path.join(source_directory, source_filename, source_filename, ""))
        source_video = read_pure_video(os.path.join(source_directory, source_filename, source_filename, "")) 
        source_video.tolist()

    print(f'Source Shape: {np.shape(source_video)}')

    if dataset != 'SCAMPS':
        # Face detection to crop
        cropped_frames = []
        face_region_all = []

        # First, compute the median bounding box across all frames
        for frame in source_video:
            if dataset == 'PURE':
                face_box = face_detection(frame, True, 1.7) # PURE
            else:
                face_box = face_detection(frame, True, 2.0) # MAUBFC and others
            face_region_all.append(face_box)
        face_region_all = np.asarray(face_region_all, dtype='int')
        face_region_median = np.median(face_region_all, axis=0).astype('int')

        # Apply the median bounding box for cropping and subsequent resizing
        for frame in source_video:
            cropped_frame = frame[int(face_region_median[1]):int(face_region_median[1]+face_region_median[3]),
                                int(face_region_median[0]):int(face_region_median[0]+face_region_median[2])]
            resized_frame = resize_to_original(cropped_frame, np.shape(source_video)[2], np.shape(source_video)[1])
            cropped_frames.append(resized_frame)

        source_video = cropped_frames

        print(f'Cropped Source Shape: {np.shape(source_video)}')

    #Randomize the driving list sequence
    driving_path = np.random.choice(driving_list, 1)[0]
    
    driving_filename = os.fsdecode(driving_path)    

    source_video_name = os.path.splitext(source_filename)[0]
    driving_video_name = os.path.splitext(driving_filename)[0]

    if dataset == 'SCAMPS':
        filename = source_video_name + '_' + driving_video_name + '.mat'

        while os.path.exists(os.path.join(opt.augmented_path, filename)) == True:
            driving_path = np.random.choice(driving_list, 1)[0]
            driving_filename = os.fsdecode(driving_path)
            driving_video_name = os.path.splitext(driving_filename)[0]
            filename = source_video_name + '_' + driving_video_name + '.mat'
    elif dataset == 'UBFC-rPPG':
        filename = source_video_name + '_' + driving_video_name + '.npy'

        while os.path.exists(os.path.join(opt.augmented_path, filename)) == True:
            driving_path = np.random.choice(driving_list, 1)[0]
            driving_filename = os.fsdecode(driving_path)
            driving_video_name = os.path.splitext(driving_filename)[0]
            filename = source_video_name + '_' + driving_video_name + '.npy'
    elif dataset == 'UBFC-PHYS':
        filename = source_video_name + '_' + driving_video_name + '.npy'

        while os.path.exists(os.path.join(opt.augmented_path, filename)) == True:
            driving_path = np.random.choice(driving_list, 1)[0]
            driving_filename = os.fsdecode(driving_path)
            driving_video_name = os.path.splitext(driving_filename)[0]
            filename = source_video_name + '_' + driving_video_name + '.npy'
    elif dataset == 'PURE':
        filename = source_video_name + '_' + driving_video_name + '.npy'

        while os.path.exists(os.path.join(opt.augmented_path, filename)) == True:
            driving_path = np.random.choice(driving_list, 1)[0]
            driving_filename = os.fsdecode(driving_path)
            driving_video_name = os.path.splitext(driving_filename)[0]
            filename = source_video_name + '_' + driving_video_name + '.npy'

    try:
        reader = imageio.get_reader(os.path.join(driving_directory, driving_filename))
    except ValueError:
        print("Unable to get driving video!")
    print("driving: ",os.path.join(driving_directory, driving_filename))
    fps = reader.get_meta_data()['fps']
    driving_video = []

    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    print("Driving shape: ", np.shape(driving_video))

    if(np.shape(driving_video)[0] < np.shape(source_video)[0]):
    #Make total frames used the same
        source_length = len(source_video)
        driving_length = len(driving_video)
        if source_length > driving_length:
            to_add = source_length - driving_length
            reversed_driving = driving_video[::-1]
            while to_add>0:
                if to_add < driving_length:
                    driving_video = np.vstack([driving_video,reversed_driving[:to_add]])
                    to_add = -1
                else:
                    driving_video = np.vstack([driving_video,reversed_driving])
                    reversed_driving = reversed_driving[::-1]
                    to_add -= driving_length
            print("Finishing resizing")

    name = source_filename + "_" + driving_filename 
    make_video(dataset, gpu, opt, source_video, driving_video,generator,kp_detector,source_directory, source_filename, driving_filename, opt.augmented_path)
    return

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

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--config", default='config/vox-256.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='', help="path to checkpoint to restore")

    # Args when processing just a single video, and not a dataset
    parser.add_argument("--source_image", default='', help="path to source image")
    parser.add_argument("--driving_video", default='', help="path to driving video")
    parser.add_argument("--result_video", default='result.mp4', help="path to output")

    parser.add_argument("--gen", default="spade", choices=["original", "spade"])
 
    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    parser.add_argument("--free_view", dest="free_view", action="store_true", help="control head pose")
    parser.add_argument("--yaw", dest="yaw", type=int, default=None, help="yaw")
    parser.add_argument("--pitch", dest="pitch", type=int, default=None, help="pitch")
    parser.add_argument("--roll", dest="roll", type=int, default=None, help="roll")
    parser.add_argument("--scamps_source", default='', help="path for scamps source")
    parser.add_argument("--augmented_path", default='', help="path for saving augmented SCAMPS videos")
    parser.add_argument("--source_path", default='', help="path for source SCAMPS videos")
    parser.add_argument("--driving_path", default='', help="path for driving videos")
    parser.add_argument("--dataset", default='UBFC-rPPG', choices=["SCAMPS", "UBFC-rPPG", "UBFC-PHYS", "PURE"], help="dataset specification")
    parser.add_argument("--mp", dest="mp", action="store_true", help="use multiprocessing")
 

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)
    parser.set_defaults(free_view=False)

    opt = parser.parse_args()

    try:
        set_start_method('spawn')
    except RuntimeError:
        print("Error! Unable to set start method to spawn.")

    source_directory = opt.source_path
    driving_directory = opt.driving_path
    augmented_directory = opt.augmented_path
    
    # Load checkpoints
    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)
    print("Checkpoints loaded!")

    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    estimate_jacobian = config['model_params']['common_params']['estimate_jacobian']

    #Listing driving video list
    driving_list = os.listdir(driving_directory)

    #Listing source video list
    source_list = sorted(os.listdir(source_directory))
    
    copy_folder(source_directory, augmented_directory)

    augmented_list = sorted(os.listdir(augmented_directory))
    # print(augmented_list)

    file_num = len(source_list)
    choose_range = range(0, file_num)
    pbar = tqdm(list(choose_range))

    # shared data resource
    p_list = []
    running_num = 0

    # Get the available GPU count
    gpu_count = torch.cuda.device_count()
    print(f'{gpu_count} GPUs are available!')

    if opt.mp is False:
        # Single process
        for i in choose_range:
            gpu_num = running_num % gpu_count
            augment_motion(opt.dataset, gpu_num, source_list, driving_list, augmented_list, i, opt, running_num, source_directory, driving_directory, generator, kp_detector)
            pbar.update(1)
        pbar.close()
    elif opt.mp is True:
        # Use multiprocessing
        print("Multiprocessing is being used!")
        for i in choose_range:
            process_flag = True
            while process_flag:  # ensure that every i creates a process
                if running_num < 5:  # in case of too many processes
                    # assign an available GPU to the process
                    gpu_num = running_num % gpu_count

                    p = Process(target=augment_motion, \
                                args=(opt.dataset, gpu_num, source_list, driving_list, augmented_list, i, opt, running_num, source_directory, driving_directory, generator, kp_detector))
                    p.start()
                    p_list.append(p)
                    running_num += 1
                    process_flag = False
                for p_ in p_list:
                    if not p_.is_alive():
                        p_list.remove(p_)
                        p_.join()
                        running_num -= 1
                        pbar.update(1)
            time.sleep(60 * 3)
        # join all processes
        for p_ in p_list:
            p_.join()
            pbar.update(1)
        pbar.close()
