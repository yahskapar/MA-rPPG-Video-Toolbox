counter=1
until [ $counter -gt 4 ]
do
echo $counter
CUDA_VISIBLE_DEVICES=0 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/scampsdataset/SCAMPS_700_gen/bin_14 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp_mod_int --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MASCAMPS_2800_MOD_INT &
sleep 1m
CUDA_VISIBLE_DEVICES=1 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/scampsdataset/SCAMPS_700_gen/bin_15 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp_mod_int --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MASCAMPS_2800_MOD_INT &
sleep 1m
CUDA_VISIBLE_DEVICES=2 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/scampsdataset/SCAMPS_700_gen/bin_16 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp_mod_int --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MASCAMPS_2800_MOD_INT &
sleep 1m
CUDA_VISIBLE_DEVICES=3 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/scampsdataset/SCAMPS_700_gen/bin_17 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp_mod_int --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MASCAMPS_2800_MOD_INT &

sleep 1m
CUDA_VISIBLE_DEVICES=0 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/scampsdataset/SCAMPS_700_gen/bin_18 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp_mod_int --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MASCAMPS_2800_MOD_INT &
sleep 1m
CUDA_VISIBLE_DEVICES=1 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/scampsdataset/SCAMPS_700_gen/bin_19 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp_mod_int --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MASCAMPS_2800_MOD_INT &
sleep 1m
CUDA_VISIBLE_DEVICES=2 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/scampsdataset/SCAMPS_700_gen/bin_20 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp_mod_int --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MASCAMPS_2800_MOD_INT &
sleep 1m

CUDA_VISIBLE_DEVICES=3 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/scampsdataset/SCAMPS_700_gen/bin_21 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp_mod_int --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MASCAMPS_2800_MOD_INT &
sleep 1m
CUDA_VISIBLE_DEVICES=0 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/scampsdataset/SCAMPS_700_gen/bin_22 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp_mod_int --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MASCAMPS_2800_MOD_INT &
sleep 1m
CUDA_VISIBLE_DEVICES=1 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/scampsdataset/SCAMPS_700_gen/bin_23 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp_mod_int --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MASCAMPS_2800_MOD_INT &
sleep 1m
CUDA_VISIBLE_DEVICES=2 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/scampsdataset/SCAMPS_700_gen/bin_24 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp_mod_int --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MASCAMPS_2800_MOD_INT &

sleep 1m
CUDA_VISIBLE_DEVICES=3 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/scampsdataset/SCAMPS_700_gen/bin_25 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp_mod_int --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MASCAMPS_2800_MOD_INT &
sleep 1m
CUDA_VISIBLE_DEVICES=0 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/scampsdataset/SCAMPS_700_gen/bin_26 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp_mod_int --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MASCAMPS_2800_MOD_INT &
sleep 1m
CUDA_VISIBLE_DEVICES=1 python augment_videos.py --config config/vox-256-spade.yaml --checkpoint checkpoints/00000189-checkpoint.pth.tar --source_path /playpen-nas-hdd/UNC_Google_Physio/scampsdataset/SCAMPS_700_gen/bin_27 --driving_path /playpen-temp/users/akshay/motion_analysis/samples_500_category/combined_hp_mod_int --relative --adapt_scale --augmented_path /playpen-temp/users/akshay/datasets/MASCAMPS_2800_MOD_INT &
wait
((counter++))
done