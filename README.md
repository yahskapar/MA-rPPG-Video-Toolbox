# Motion Matters: Neural Motion Transfer for Better Camera Physiological Sensing

[Pre-print](https://arxiv.org/abs/2303.12059) | [Project Website](https://motion-matters.github.io/)

![Examples of motion augmentation applied to subjects in the UBFC-rPPG dataset.](./assets/ma_rppg_video_toolbox_teaser.gif)

## Abstract

Machine learning models for camera-based physiological measurement can have weak generalization due to a lack of representative training data. Body motion is one of the most significant sources of noise when attempting to recover the subtle cardiac pulse from a video. We explore motion transfer as a form of data augmentation to introduce motion variation while preserving physiological changes. We adapt a neural video synthesis approach to augment videos for the task of remote photoplethysmography (PPG) and study the effects of motion augmentation with respect to 1. the magnitude and 2. the type of motion. After training on motion-augmented versions of publicly available datasets, the presented inter-dataset results on five benchmark datasets show improvements of up to 75% over existing state-of-the-art results. Our findings illustrate the utility of motion transfer as a data augmentation technique for improving the generalization of models for camera-based physiological sensing. We release our code and pre-trained models for using motion transfer as a data augmentation technique.

# Setup

STEP1: `bash setup.sh` 

STEP2: `conda activate ma-rppg-video-toolbox` 

STEP3: `pip install -r requirements.txt`

STEP 4: Install PyTorch using the below command,

```
pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
```

STEP 5: Download the appropriate [pre-trained model](#pretrained-models) and place it in the appropriate folder within `checkpoints/`. We recommend using Vox-256-New over Vox-256-Beta.

# Usage

This motion-augmentation pipeline supports three rPPG video datasets - UBFC-rPPG, PURE, and SCAMPS. The pipeline utilizes an unofficial pytorch implementation of the paper [One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing](https://nvlabs.github.io/face-vid2vid/) by Ting-Chun Wang, Arun Mallya, and Ming-Yu Liu (NVIDIA). A link to the original, unofficial implementation can be found [here](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis). `Python 3.6.13` and `Pytorch 1.8.2` are used.

Below is a basic example of utilizing `augment_videos.py` to augment all subjects provided in the UBFC-rPPG with motion based on a supplied directory of driving videos and the Vox-256-New pre-trained model:
```
python augment_videos.py --config ./checkpoints/checkpoint_new/vox-256-spade.yaml --checkpoint ./checkpoints/checkpoint_new/00000189-checkpoint.pth.tar --source_path /path/to/source/dataset/folder --driving_path /path/to/driving/dataset/folder --augmented_path /path/to/augmented/dataset/folder/to/generate --relative --adapt_scale --dataset UBFC-rPPG
```
Note that an augmented output path is specified with `--augmented_path`. The augmented output includes the exact same folder structure as the input dataset (e.g., UBFC-rPPG dataset's DATASET2 folder structure), and contains all  of the corresponding ground truth files that are copied over.

A naive implementation of multiprocessing can be enabled with the `--mp` command line option to speed-up the motion augmentation pipeline. Depending on your computing environment, this is not recommended. Multiprocessing support will be refined in a future update to this code repository.

Additionally, we provide motion analysis scripts in the `motion_analysis` folder to generate and analyze videos processed using OpenFace. Please refer to the [OpenFace GitHub repo](https://github.com/TadasBaltrusaitis/OpenFace) for instructions on how to properly install OpenFace. We also provide pre-trained models in `pretrained_models` and example configs and dataloaders in `MA_training` for use with the [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox).

# Pretrained Models

The below pre-trained models were obtained from [here]([here](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis)).

  Model  |  Train Set   | Baidu Netdisk | Media Fire | 
 ------- |------------  |-----------    |--------      |
 Vox-256-Beta| VoxCeleb-v1  | [Baidu](https://pan.baidu.com/s/1lLS4ArbK2yWelsL-EtwU8g) (PW: c0tc)|  [MF](https://www.mediafire.com/folder/rw51an7tk7bh2/TalkingHead)  |
 Vox-256-New | VoxCeleb-v1  |  -  |  [MF](https://www.mediafire.com/folder/fcvtkn21j57bb/TalkingHead_Update)  |
 Vox-512 | VoxCeleb-v2  |  soon  |  soon  |

# Acknowledgments
Thanks to [zhanglonghao1992](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis), [NV](https://github.com/NVlabs/face-vid2vid), [AliaksandrSiarohin](https://github.com/AliaksandrSiarohin/first-order-model), and [DeepHeadPose](https://github.com/DriverDistraction/DeepHeadPose) for their useful code implementations!

# Citation
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
