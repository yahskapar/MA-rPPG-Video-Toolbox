# This code is related to a WACV 2024 paper: https://motion-matters.github.io/
# Please refer to the project website and GitHub README for more details.

import os, gc, yaml
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

import torch
from torch.multiprocessing import Queue, Pool, Manager, set_start_method, cpu_count
from threading import Thread

from animate import make_animation
from utils import copy_folder, load_checkpoints, read_ubfc_video, read_pure_video, read_scamps_video, save_scamps_video
from face_detection import resize_to_original, face_detection
from skimage.transform import resize
import imageio
import warnings
warnings.filterwarnings("ignore")

def make_video(dataset, opt, source_video, driving_video,generator,kp_detector,he_estimator,estimate_jacobian,source_directory, source_filename, driving_filename, augmented_path):
    final_preds = []

    # TODO: The progress bar will effectively be broken when multi-processing is used
    # A fix will be implemented in a future update to this toolbox. Uncomment the below 
    # line and its usage elseswhere in this function if you want per-frame progress updates.
    # frames_pbar = tqdm(list(range(min(np.shape(source_video)[0], np.shape(driving_video)[0]))))

    for frames in range(min(np.shape(source_video)[0], np.shape(driving_video)[0])):
        source_image = resize(source_video[frames], (256, 256))[..., :3]
        # TODO: Perform batch processing
        predictions = make_animation(source_image, driving_video, frames, generator, kp_detector, he_estimator, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, estimate_jacobian=estimate_jacobian, cpu=opt.cpu, free_view=opt.free_view, yaw=opt.yaw, pitch=opt.pitch, roll=opt.roll)
        final_preds.append(predictions)
    #     frames_pbar.update(1)
    # frames_pbar.close()
    np_preds = np.squeeze(np.asarray(final_preds))

    if dataset == 'SCAMPS':
        final_preds = [resize(frame, (240, 240))[..., :3] for frame in np_preds]
        save_scamps_video(source_directory, source_filename, driving_filename, final_preds, augmented_path)
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

    # Cleanup
    del final_preds, np_preds

    return

def augment_motion(dataset, source_list, driving_list, i, opt, source_directory, driving_directory, generator, kp_detector, he_estimator, estimate_jacobian):
    # TODO: Improve error handling throughout this function
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
        # TODO: Add config options for this
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
    driving_video = []

    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    print("Driving shape: ", np.shape(driving_video))

    # Make total frames used the same
    if(np.shape(driving_video)[0] < np.shape(source_video)[0]):
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

    make_video(dataset, opt, source_video, driving_video, generator, kp_detector, he_estimator, estimate_jacobian, source_directory, source_filename, driving_filename, opt.augmented_path)
    return

def worker(args):
    dataset, source_list, driving_list, i, opt, source_directory, driving_directory, generator, kp_detector, he_estimator, estimate_jacobian, gpu_queue, progress_queue = args

    gpu_num = gpu_queue.get()  # Get a GPU ID from the queue
    torch.cuda.set_device(gpu_num)

    # Now perform the task with the given GPU
    augment_motion(dataset, source_list, driving_list, i, opt, source_directory, driving_directory, generator, kp_detector, he_estimator, estimate_jacobian)

    # Clean-up
    gc.collect()
    torch.cuda.empty_cache()

    gpu_queue.put(gpu_num)  # Put the GPU ID back into the queue
    progress_queue.put(1)


def process_progress_updates(progress_queue, total_tasks, pbar):
    completed_tasks = 0
    while completed_tasks < total_tasks:
        completed_tasks += progress_queue.get()
        pbar.update(1)
    pbar.close()

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
        set_start_method('spawn', force=True)
    except RuntimeError:
        print("Error! Unable to set start method to spawn.")

    source_directory = opt.source_path
    driving_directory = opt.driving_path
    augmented_directory = opt.augmented_path
    
    # Load checkpoints
    generator, kp_detector, he_estimator = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, gen=opt.gen, cpu=opt.cpu)
    print("Checkpoints loaded!")

    # Load config
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    estimate_jacobian = config['model_params']['common_params']['estimate_jacobian']

    # Get driving video list
    driving_list = os.listdir(driving_directory)

    # Get source video list
    source_list = sorted(os.listdir(source_directory))
    
    copy_folder(source_directory, augmented_directory)

    file_num = len(source_list)
    choose_range = range(0, file_num)
    pbar = tqdm(list(choose_range))

    # Get the available GPU count
    gpu_count = torch.cuda.device_count()
    print(f'{gpu_count} GPUs are available!')

    if opt.mp is False:
        print("Multiprocessing is NOT being used. Please consider enabling it with --mp.")
        for i in choose_range:
            augment_motion(opt.dataset, source_list, driving_list, i, opt, source_directory, driving_directory, generator, kp_detector, he_estimator, estimate_jacobian)
            pbar.update(1)
        pbar.close()
    elif opt.mp is True:
        print("Multiprocessing is being used!")

        with Manager() as manager:
            gpu_queue = manager.Queue()
            progress_queue = manager.Queue()

            # Initialize GPU queue with available GPU IDs
            for gpu_id in range(torch.cuda.device_count()):
                gpu_queue.put(gpu_id)

            # Prepare arguments for each task
            tasks = [(opt.dataset, source_list, driving_list, i, opt, source_directory, driving_directory, generator, kp_detector, he_estimator, estimate_jacobian, gpu_queue, progress_queue) for i, _ in enumerate(choose_range)]

            num_processes = min(cpu_count(), torch.cuda.device_count())

            # Start the thread for processing progress updates
            progress_thread = Thread(target=process_progress_updates, args=(progress_queue, len(tasks), pbar))
            progress_thread.start()

            # Create a Pool and distribute the tasks
            pool = Pool(processes=min(cpu_count(), torch.cuda.device_count()))

            # Using imap_unordered for potentially more efficient task distribution
            for _ in pool.imap_unordered(worker, tasks):
                pass

            # Close the pool and wait for all worker processes to finish
            pool.close()
            pool.join()

            progress_thread.join()
