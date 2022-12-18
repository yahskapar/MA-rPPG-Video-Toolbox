CUDA_VISIBLE_DEVICES=0 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/UBFC-rPPG/DATASET_2/bin_0 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MAUBFC &
sleep 1m
CUDA_VISIBLE_DEVICES=1 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/UBFC-rPPG/DATASET_2/bin_1 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MAUBFC &
sleep 1m
CUDA_VISIBLE_DEVICES=2 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/UBFC-rPPG/DATASET_2/bin_2 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MAUBFC &
sleep 1m
CUDA_VISIBLE_DEVICES=3 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/UBFC-rPPG/DATASET_2/bin_3 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MAUBFC &

sleep 1m
CUDA_VISIBLE_DEVICES=0 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/UBFC-rPPG/DATASET_2/bin_4 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MAUBFC &
sleep 1m
CUDA_VISIBLE_DEVICES=1 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/UBFC-rPPG/DATASET_2/bin_5 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MAUBFC &
sleep 1m
CUDA_VISIBLE_DEVICES=2 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/UBFC-rPPG/DATASET_2/bin_6 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MAUBFC &

wait