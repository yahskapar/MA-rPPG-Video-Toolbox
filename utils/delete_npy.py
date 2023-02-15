import os

source_dir = "/playpen-nas-ssd/akshay/UNC_Google_Physio/datasets/MAUBFC_PHYS_T2"

for root, dirs, files in os.walk(source_dir):
    for filename in files:
        if filename.endswith(".npy"):
            file_path = os.path.join(root, filename)
            os.remove(file_path)
