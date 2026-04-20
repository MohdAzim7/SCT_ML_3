import os
import shutil

source_folder = "datasets/train/train"
cat_folder = os.path.join(source_folder, "cats")
dog_folder = os.path.join(source_folder, "dogs")

os.makedirs(cat_folder, exist_ok=True)
os.makedirs(dog_folder, exist_ok=True)

for file in os.listdir(source_folder):

    # skip folders
    if os.path.isdir(os.path.join(source_folder, file)):
        continue

    src = os.path.join(source_folder, file)

    if file.startswith("cat"):
        shutil.move(src, os.path.join(cat_folder, file))

    elif file.startswith("dog"):
        shutil.move(src, os.path.join(dog_folder, file))

print(" Dataset organized successfully!")