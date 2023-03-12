import os

source_dir = "/put/your/file/path/here"

for root, dirs, files in os.walk(source_dir):
    for filename in files:
        if filename == "vid.avi":
            subfolder = os.path.basename(root)
            new_filename = subfolder + "_vid.avi"
            old_path = os.path.join(root, filename)
            new_path = os.path.join(root, new_filename)
            os.rename(old_path, new_path)