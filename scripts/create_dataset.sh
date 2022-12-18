# counter=1
# until [ $counter -gt 4 ]
# do
# echo $counter
CUDA_VISIBLE_DEVICES=0 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/scampsdataset/SCAMPS_700_gen/bin_0 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp_mod_int --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MASCAMPS_700_MOD_INT &
sleep 1m
CUDA_VISIBLE_DEVICES=1 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/scampsdataset/SCAMPS_700_gen/bin_1 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp_mod_int --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MASCAMPS_700_MOD_INT &
sleep 1m
CUDA_VISIBLE_DEVICES=2 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/scampsdataset/SCAMPS_700_gen/bin_2 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp_mod_int --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MASCAMPS_700_MOD_INT &
sleep 1m
CUDA_VISIBLE_DEVICES=3 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/scampsdataset/SCAMPS_700_gen/bin_3 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp_mod_int --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MASCAMPS_700_MOD_INT &

sleep 1m
CUDA_VISIBLE_DEVICES=0 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/scampsdataset/SCAMPS_700_gen/bin_4 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp_mod_int --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MASCAMPS_700_MOD_INT &
sleep 1m
CUDA_VISIBLE_DEVICES=1 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/scampsdataset/SCAMPS_700_gen/bin_5 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp_mod_int --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MASCAMPS_700_MOD_INT &
sleep 1m
CUDA_VISIBLE_DEVICES=2 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/scampsdataset/SCAMPS_700_gen/bin_6 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp_mod_int --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MASCAMPS_700_MOD_INT &

wait
# ((counter++))
# done