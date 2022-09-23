# MASCAMPS

TODO: Clean up artifacts from unofficial implementation by zhanglonghao1992. 

Motion-Augmented Synthetics for Camera Measurement of Physiological Signals

Akshay Paruchuri (UNC Chapel Hill), Xin Liu (Google), Daniel McDuff (Google), and Soumyadip Sengupta (UNC Chapel Hill)

This motion-augmentation technique for real and/or synthetic datasets used for training models toward camera measurement of physiological signals utilizes an unofficial pytorch implementation of the paper "One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing" by Ting-Chun Wang, Arun Mallya, and Ming-Yu Liu (NVIDIA). 

A link to the original, unofficial implementation can be found [here](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis). ```Python 3.6``` and ```Pytorch 1.7``` are used in Linux Mint 21 Cinnamon. An ```environment.yml``` is provided in the repo for easier installation of the dependencies required. You can use that environment file when creating the conda environment as follows: ```conda env create -f environment.yml```. TODO: The environment.yml file might have to be refined to be platform-agnostic somehow.

Here's a typical workflow with the current (09/20/2022) iteration of this code:

1. In a conda environment, install required dependencies using the environment.yml file as described above. Make sure you are utilizing Python 3.6 and Pytorch 1.7 as well.
2. Download a pretrained model from the below table of pretrained models. In the future, instructins will be provided to utilize a modified train.py for training a new model from scratch or training from existing pretrained models.
3. Place source videos and driving videos you wish to augment in the respective ```source``` and ```driving``` folders.
4. At this point, you can use either of the below commands to augment and analyze a single source video and driving video pair, or augment a bunch of pairs of source videos and driving videos.

Augment and analyze a single source video and driving video pair (using GPU):
```
CUDA_VISIBLE_DEVICES=0 python demo.py --config config/vox-256-spade.yaml --checkpoint checkpoints/[checkpoint_file_name] --scamps_source [path_to_scamps_source_video]  --driving_video [path_to_driving_video] --relative --adapt_scale
```

Augment pairs of source videos and driving videos (using GPU):
```
CUDA_VISIBLE_DEVICES=0 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/[checkpoint_file_name] --source_path source --driving_path driving --relative --adapt_scale --augmented_path augmented
```

You can avoid using a GPU in either of the above commands by removing the CUDA_VISIBLE_DEVICES portion of the command and adding the --cpu option to the command. 

Pretrained Model:  
--------

  Model  |  Train Set   | Baidu Netdisk | Media Fire | 
 ------- |------------  |-----------    |--------      |
 Vox-256-Beta| VoxCeleb-v1  | [Baidu](https://pan.baidu.com/s/1lLS4ArbK2yWelsL-EtwU8g) (PW: c0tc)|  [MF](https://www.mediafire.com/folder/rw51an7tk7bh2/TalkingHead)  |
 Vox-256-New | VoxCeleb-v1  |  -  |  [MF](https://www.mediafire.com/folder/fcvtkn21j57bb/TalkingHead_Update)  |
 Vox-512 | VoxCeleb-v2  |  soon  |  soon  |

 Acknowlegement: 
--------
Thanks to [zhanglonghao1992](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis), [NV](https://github.com/NVlabs/face-vid2vid), [AliaksandrSiarohin](https://github.com/AliaksandrSiarohin/first-order-model) and [DeepHeadPose](https://github.com/DriverDistraction/DeepHeadPose).
