import os

source_dir = "/put/your/file/path/here"

for root, dirs, files in os.walk(source_dir):
    for filename in files:
        if filename.endswith(".npy"):
            file_path = os.path.join(root, filename)
            os.remove(file_path)
