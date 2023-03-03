import csv
import os
import shutil
import time

src_dir = '/playpen-nas-ssd/data/datasets/TalkingHead-1KH/train/use_pose'
dst_dir = '/playpen-nas-ssd/data/datasets/TalkingHead-1KH/train/category_UBFC-PHYS_T2'
csv_file = '/playpen-nas-ssd/yulupan/more_plots/match/csv/match_ubfc_talkinghead_no_dupes.csv'

i=0
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if 'T2' in row['source']:
            video_name = row['driving']
            src_path = os.path.join(src_dir, video_name)
            dst_path = os.path.join(dst_dir, video_name)
            if os.path.exists(src_path):
                print(src_path)
                print(dst_path)
                shutil.copy(src_path, dst_path)
                i = i + 1
            else:
                print("Video does not exist in specified source directory!")
print(i)