# Motion Matters: Neural Motion Transfer for Better Camera Physiological Sensing

<p align="center">
:fire: Please remember to :star: this repo if you find it useful and <a href="https://github.com/Roni-Lab/MA-rPPG-Video-Toolbox#scroll-citation">cite</a> our work if you end up using it in your work! :fire:
</p>
<p align="center">
:fire: If you have any questions or concerns, please create an issue :memo:! :fire:
</p>

<p align="center">
<a href="https://arxiv.org/abs/2303.12059">Pre-print</a> | <a href="https://motion-matters.github.io/">Project Website</a>
</p>

<p align="center">
  <img src="./assets/ma_rppg_video_toolbox_teaser.gif" alt="Examples of motion augmentation applied to subjects in the UBFC-rPPG dataset." />
</p>

## :book: Abstract

Machine learning models for camera-based physiological measurement can have weak generalization due to a lack of representative training data. Body motion is one of the most significant sources of noise when attempting to recover the subtle cardiac pulse from a video. We explore motion transfer as a form of data augmentation to introduce motion variation while preserving physiological changes of interest. We adapt a neural video synthesis approach to augment videos for the task of remote photoplethysmography (rPPG) and study the effects of motion augmentation with respect to 1. the magnitude and 2. the type of motion. After training on motion-augmented versions of publicly available datasets, we demonstrate a 47% improvement over existing inter-dataset results using various state-of-the-art methods on the PURE dataset. We also present inter-dataset results on five benchmark datasets to show improvements of up to 79% using TS-CAN, a neural rPPG estimation method. Our findings illustrate the usefulness of motion transfer as a data augmentation technique for improving the generalization of models for camera-based physiological sensing.

## :wrench: Setup

STEP1: `bash setup.sh` 

STEP2: `conda activate ma-rppg-video-toolbox` 

STEP3: `pip install -r requirements.txt`

STEP 4: Install PyTorch using the below command,

```
pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
```
The exact versioning may vary depending on your computing environment and what GPUs you have access to.

STEP 5: Download the appropriate [pre-trained model](#sparkles-pretrained-models) and place it in the appropriate folder within `checkpoints/`. We recommend using Vox-256-New over Vox-256-Beta. You will also have to place the appropriate `.yaml` file in the `config/` folder, and make sure it's appropriately pointed to using the `--config` parameter when running `augment_videos.py`.

## :computer: Usage

This motion-augmentation pipeline supports three rPPG video datasets - UBFC-rPPG, PURE, and SCAMPS. The pipeline utilizes an unofficial pytorch implementation of the paper [One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing](https://nvlabs.github.io/face-vid2vid/) by Ting-Chun Wang, Arun Mallya, and Ming-Yu Liu (NVIDIA). A link to the original, unofficial implementation can be found [here](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis). `Python 3.6.13` and `Pytorch 1.8.2` are used.

Below is a basic example of utilizing `augment_videos.py` to augment all subjects provided in the UBFC-rPPG with motion based on a supplied directory of driving videos and the Vox-256-New pre-trained model:
```
python augment_videos.py --config ./checkpoints/checkpoint_new/vox-256-spade.yaml --checkpoint ./checkpoints/checkpoint_new/00000189-checkpoint.pth.tar --source_path /path/to/source/dataset/folder --driving_path /path/to/driving/dataset/folder --augmented_path /path/to/augmented/dataset/folder/to/generate --relative --adapt_scale --dataset UBFC-rPPG
```
Note that an augmented output path is specified with `--augmented_path`. The augmented output includes the exact same folder structure as the input dataset (e.g., UBFC-rPPG dataset's DATASET2 folder structure), and contains all  of the corresponding ground truth files that are copied over. More information about supported datasets, as well as a Google Drive link to videos that we used as driving videos for our experiments in the paper, can be found in the [datasets section](#file_folder-datasets) below.

A naive implementation of multiprocessing can be enabled with the `--mp` command line option to speed-up the motion augmentation pipeline. Depending on your computing environment, this may be the best way to generate augmented datasets using this GitHub repo. It is strongly recommend that you utilize `CUDA_VISIBLE_DEVICES` alongside the aforementioned python command when using multiprocessing (e.g., `CUDA_VISIBLE_DEVICES=0,1,2,3 python augmented_videos.py ...`). The number of processes generated will depending on the number of CPU cores or GPUs, and for the time being a single GPU should not be overloaded with multiple processes due to instability with videos of variable sizes.

You may find code for motion analysis and visualization using [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) to be useful [here](https://github.com/ubicomplab/rPPG-Toolbox/tree/main/tools/motion_analysis), in the [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) repo. We also provide pre-trained models in `pretrained_models/` for use with the [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox). You can train with your own, motion-augmented data by following the instructions in the [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) repo [here](https://github.com/ubicomplab/rPPG-Toolbox#blue_book-motion-augmented-training).

## :sparkles: Pretrained Models

The below pre-trained models were obtained from [here](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis).

  Model  |  Train Set   | Baidu Netdisk | Media Fire | 
 ------- |------------  |-----------    |--------      |
 Vox-256-Beta| VoxCeleb-v1  | [Baidu](https://pan.baidu.com/s/1lLS4ArbK2yWelsL-EtwU8g) (PW: c0tc)|  [MF](https://www.mediafire.com/folder/rw51an7tk7bh2/TalkingHead)  |
 Vox-256-New | VoxCeleb-v1  |  -  |  [MF](https://www.mediafire.com/folder/fcvtkn21j57bb/TalkingHead_Update)  |
 Vox-512 | VoxCeleb-v2  |  soon  |  soon  |

 A Google Drive back-up of these pre-trained models (in the form of a `checkpoints/` folder that can be directly used in this repo) can be found [here](https://drive.google.com/drive/folders/1knacMCP3hhS49wsZ7xNVlsU1sZCpr1-0?usp=sharing).

 ## :file_folder: Datasets

 The following source rPPG video datasets can be downloaded via the below links (note that aside from SCAMPS, these datasets require some sort of interaction with the authors and/or a filled out form):
 * [UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg)
 * [PURE](https://www.tu-ilmenau.de/en/university/departments/department-of-computer-science-and-automation/profile/institutes-and-groups/institute-of-computer-and-systems-engineering/group-for-neuroinformatics-and-cognitive-robotics/data-sets-code/pulse-rate-detection-dataset-pure)
 * [SCAMPS](https://github.com/danmcduff/scampsdataset)

 The [TalkingHead-1KH](https://github.com/deepimagination/TalkingHead-1KH) driving videos we used, filtered using either the mean standard deviation in head pose rotations or the mean standard deviation in facial AUs as described in our paper, can be found [here](https://drive.google.com/drive/folders/1aH7RqpxvsfkvY8v7lHxG_U1dG_ZNKgcf?usp=sharing). In addition to [citing our work](#scroll-citation) if you use this motion augmentation pipeline, please [cite the TalkingHead-1KH dataset](https://github.com/deepimagination/TalkingHead-1KH#citation) if you use the aforementioned driving videos. Our self-captured, constrained driving video set (CDVS) as described in the paper will be released soon.

## :scroll: Acknowledgments
Thanks to [zhanglonghao1992](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis), [NV](https://github.com/NVlabs/face-vid2vid), [AliaksandrSiarohin](https://github.com/AliaksandrSiarohin/first-order-model), and [DeepHeadPose](https://github.com/DriverDistraction/DeepHeadPose) for their useful code implementations!

## :scroll: Citation
If you find our [paper](https://arxiv.org/abs/2303.12059) or this toolbox useful for your research, please cite our work.

```
@misc{paruchuri2023motion,
      title={Motion Matters: Neural Motion Transfer for Better Camera Physiological Sensing}, 
      author={Akshay Paruchuri and Xin Liu and Yulu Pan and Shwetak Patel and Daniel McDuff and Soumyadip Sengupta},
      year={2023},
      eprint={2303.12059},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
