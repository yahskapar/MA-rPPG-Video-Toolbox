# Motion Matters: Neural Motion Transfer for Better Camera Physiological Sensing

## Abstract
Machine learning models for camera-based physiological measurement can have weak generalization due to a lack of representative training data. Body motion is one of the most significant sources of noise when attempting to recover the subtle cardiac pulse from a video. We explore motion transfer as a form of data augmentation to introduce motion variation while preserving physiological changes. We adapt a neural video synthesis approach to augment videos for the task of remote photoplethysmography (PPG) and study the effects of motion augmentation with respect to 1. the magnitude and 2. the type of motion. After training on motion-augmented versions of publicly available datasets, the presented inter-dataset results on five benchmark datasets show improvements of up to 75% over existing state-of-the-art results. Our findings illustrate the utility of motion transfer as a data augmentation technique for improving the generalization of models for camera-based physiological sensing. We release our code and pre-trained models for using motion transfer as a data augmentation technique.

## Code Description

This motion-augmentation technique for real and/or synthetic datasets used for training models toward camera measurement of physiological signals utilizes an unofficial pytorch implementation of the paper "One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing" by Ting-Chun Wang, Arun Mallya, and Ming-Yu Liu (NVIDIA). A link to the original, unofficial implementation can be found [here](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis). ```Python 3.6``` and ```Pytorch 1.7``` are used in Linux Mint 21 Cinnamon. An ```environment.yml``` is provided in the repo for easier installation of the dependencies required. You can use that environment file when creating the conda environment as follows: ```conda env create -f environment.yml```.

Here's a typical workflow with the current (03/13/2023) iteration of this code:

1. In a conda environment, install required dependencies using the environment.yml file as described above. Make sure you are utilizing Python 3.6 and Pytorch 1.7 or higher as well.
2. Download a pretrained model from the below table of pretrained models. In the future, instructins will be provided to utilize a modified train.py for training a new model from scratch or training from existing pretrained models.
3. Augment pairs of source videos and driving videos (using GPU):
```
CUDA_VISIBLE_DEVICES=0 python augment_videos.py --config ./checkpoints/checkpoint_new/vox-256-spade.yaml --checkpoint ./checkpoints/checkpoint_new/00000189-checkpoint.pth.tar --source_path /path/to/source/dataset/folder --driving_path /path/to/driving/dataset/folder --augmented_path /path/to/augmented/dataset/folder/to/generate --relative --adapt_scale --dataset UBFC-rPPG
```

The currently supported datasets for augmentation are UBFC-rPPG, PURE, and SCAMPS.

Additionally, we provide motion analysis scripts in the `motion_analysis` folder to generate and analyze videos processed using OpenFace. Please refer to the [OpenFace GitHub repo](https://github.com/TadasBaltrusaitis/OpenFace) for instructions on how to properly install OpenFace. We also provide pre-trained models in `pretrained_models` and example configs and dataloaders in `MA_training` for use with the [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox).

Pretrained Model:  
--------

  Model  |  Train Set   | Baidu Netdisk | Media Fire | 
 ------- |------------  |-----------    |--------      |
 Vox-256-Beta| VoxCeleb-v1  | [Baidu](https://pan.baidu.com/s/1lLS4ArbK2yWelsL-EtwU8g) (PW: c0tc)|  [MF](https://www.mediafire.com/folder/rw51an7tk7bh2/TalkingHead)  |
 Vox-256-New | VoxCeleb-v1  |  -  |  [MF](https://www.mediafire.com/folder/fcvtkn21j57bb/TalkingHead_Update)  |
 Vox-512 | VoxCeleb-v2  |  soon  |  soon  |

 Acknowlegement: 
--------
Thanks to [zhanglonghao1992](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis), [NV](https://github.com/NVlabs/face-vid2vid), [AliaksandrSiarohin](https://github.com/AliaksandrSiarohin/first-order-model), and [DeepHeadPose](https://github.com/DriverDistraction/DeepHeadPose) for their useful code implementations!
