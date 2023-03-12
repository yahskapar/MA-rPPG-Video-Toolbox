import csv
import os
import shutil
import time

src_dir = "/put/your/file/path/here"
dst_dir = "/put/your/file/path/here"
csv_file = "/put/your/file/path/here"

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