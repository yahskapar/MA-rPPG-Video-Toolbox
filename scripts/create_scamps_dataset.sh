CUDA_VISIBLE_DEVICES=0 python augment_videos.py --config ./checkpoints/checkpoint_new/vox-256-spade.yaml --checkpoint ./checkpoints/checkpoint_new/00000189-checkpoint.pth.tar --source_path /playpen-nas-ssd/tarbucket/users/akshay/datasets/SCAMPS_2800/1st_200/d1 --driving_path /playpen-nas-ssd/data/datasets/TalkingHead-1KH/train/new_categories/all_au_pool/new_au_moderate_picked --augmented_path /playpen-nas-ssd/akshay/UNC_Google_Physio/datasets/MASCAMPS_MED_AU --relative --adapt_scale --dataset SCAMPS &
sleep 4m

CUDA_VISIBLE_DEVICES=1 python augment_videos.py --config ./checkpoints/checkpoint_new/vox-256-spade.yaml --checkpoint ./checkpoints/checkpoint_new/00000189-checkpoint.pth.tar --source_path /playpen-nas-ssd/tarbucket/users/akshay/datasets/SCAMPS_2800/1st_200/d2 --driving_path /playpen-nas-ssd/data/datasets/TalkingHead-1KH/train/new_categories/all_au_pool/new_au_moderate_picked --augmented_path /playpen-nas-ssd/akshay/UNC_Google_Physio/datasets/MASCAMPS_MED_AU --relative --adapt_scale --dataset SCAMPS &
sleep 4m

CUDA_VISIBLE_DEVICES=2 python augment_videos.py --config ./checkpoints/checkpoint_new/vox-256-spade.yaml --checkpoint ./checkpoints/checkpoint_new/00000189-checkpoint.pth.tar --source_path /playpen-nas-ssd/tarbucket/users/akshay/datasets/SCAMPS_2800/1st_200/d3 --driving_path /playpen-nas-ssd/data/datasets/TalkingHead-1KH/train/new_categories/all_au_pool/new_au_moderate_picked --augmented_path /playpen-nas-ssd/akshay/UNC_Google_Physio/datasets/MASCAMPS_MED_AU --relative --adapt_scale --dataset SCAMPS &
sleep 4m

CUDA_VISIBLE_DEVICES=3 python augment_videos.py --config ./checkpoints/checkpoint_new/vox-256-spade.yaml --checkpoint ./checkpoints/checkpoint_new/00000189-checkpoint.pth.tar --source_path /playpen-nas-ssd/tarbucket/users/akshay/datasets/SCAMPS_2800/1st_200/d4 --driving_path /playpen-nas-ssd/data/datasets/TalkingHead-1KH/train/new_categories/all_au_pool/new_au_moderate_picked --augmented_path /playpen-nas-ssd/akshay/UNC_Google_Physio/datasets/MASCAMPS_MED_AU --relative --adapt_scale --dataset SCAMPS &
sleep 4m

CUDA_VISIBLE_DEVICES=4 python augment_videos.py --config ./checkpoints/checkpoint_new/vox-256-spade.yaml --checkpoint ./checkpoints/checkpoint_new/00000189-checkpoint.pth.tar --source_path /playpen-nas-ssd/tarbucket/users/akshay/datasets/SCAMPS_2800/1st_200/d5 --driving_path /playpen-nas-ssd/data/datasets/TalkingHead-1KH/train/new_categories/all_au_pool/new_au_moderate_picked --augmented_path /playpen-nas-ssd/akshay/UNC_Google_Physio/datasets/MASCAMPS_MED_AU --relative --adapt_scale --dataset SCAMPS &
sleep 4m

CUDA_VISIBLE_DEVICES=5 python augment_videos.py --config ./checkpoints/checkpoint_new/vox-256-spade.yaml --checkpoint ./checkpoints/checkpoint_new/00000189-checkpoint.pth.tar --source_path /playpen-nas-ssd/tarbucket/users/akshay/datasets/SCAMPS_2800/1st_200/d6 --driving_path /playpen-nas-ssd/data/datasets/TalkingHead-1KH/train/new_categories/all_au_pool/new_au_moderate_picked --augmented_path /playpen-nas-ssd/akshay/UNC_Google_Physio/datasets/MASCAMPS_MED_AU --relative --adapt_scale --dataset SCAMPS &
sleep 4m

CUDA_VISIBLE_DEVICES=6 python augment_videos.py --config ./checkpoints/checkpoint_new/vox-256-spade.yaml --checkpoint ./checkpoints/checkpoint_new/00000189-checkpoint.pth.tar --source_path /playpen-nas-ssd/tarbucket/users/akshay/datasets/SCAMPS_2800/1st_200/d7 --driving_path /playpen-nas-ssd/data/datasets/TalkingHead-1KH/train/new_categories/all_au_pool/new_au_moderate_picked --augmented_path /playpen-nas-ssd/akshay/UNC_Google_Physio/datasets/MASCAMPS_MED_AU --relative --adapt_scale --dataset SCAMPS &
sleep 4m

CUDA_VISIBLE_DEVICES=7 python augment_videos.py --config ./checkpoints/checkpoint_new/vox-256-spade.yaml --checkpoint ./checkpoints/checkpoint_new/00000189-checkpoint.pth.tar --source_path /playpen-nas-ssd/tarbucket/users/akshay/datasets/SCAMPS_2800/1st_200/d8 --driving_path /playpen-nas-ssd/data/datasets/TalkingHead-1KH/train/new_categories/all_au_pool/new_au_moderate_picked --augmented_path /playpen-nas-ssd/akshay/UNC_Google_Physio/datasets/MASCAMPS_MED_AU --relative --adapt_scale --dataset SCAMPS &
sleep 4m

wait