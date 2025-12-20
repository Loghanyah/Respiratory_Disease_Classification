import os
import shutil
import random
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------
ORIGINAL_DATA = "/Volumes/rloghanyah/PSM/Gammatonegram/PNG/Images"
OUTPUT_DIR = "/Volumes/rloghanyah/PSM/Dataset_Split_New/PNG/Exp2-Normal_vs_Abnormal"
SPLIT = (0.7, 0.15, 0.15)  # train, val, test

NORMAL_CLASS = "Normal"
ABNORMAL_CLASSES = ["Asthma", "BRON_2_Diseases", "COPD", "HF", "Pneumonia"]

# ----------------------------
# PREPARE OUTPUT DIRECTORIES
# ----------------------------
def make_dirs():
    for split in ["train", "val", "test"]:
        for cls in ["Normal", "Abnormal"]:
            path = os.path.join(OUTPUT_DIR, split, cls)
            os.makedirs(path, exist_ok=True)
    print("✔ Folder structure created.")

# ----------------------------
# LOAD FILES
# ----------------------------
def load_images():
    normal_images = [os.path.join(ORIGINAL_DATA, NORMAL_CLASS, f)
                     for f in os.listdir(os.path.join(ORIGINAL_DATA, NORMAL_CLASS))
                     if f.lower().endswith((".png",".jpg",".jpeg",".webp",".avif"))]

    abnormal_images = []
    for cls in ABNORMAL_CLASSES:
        cls_path = os.path.join(ORIGINAL_DATA, cls)
        imgs = [os.path.join(cls_path, f)
                for f in os.listdir(cls_path)
                if f.lower().endswith((".png",".jpg",".jpeg",".webp",".avif"))]
        abnormal_images.extend(imgs)

    print("Normal count:", len(normal_images))
    print("Abnormal count:", len(abnormal_images))

    return normal_images, abnormal_images

# ----------------------------
# SPLIT DATA
# ----------------------------
def split_data(file_list):
    random.shuffle(file_list)
    n = len(file_list)
    train_end = int(SPLIT[0] * n)
    val_end = train_end + int(SPLIT[1] * n)
    return file_list[:train_end], file_list[train_end:val_end], file_list[val_end:]

# ----------------------------
# SAVE TO FOLDERS
# ----------------------------
def copy_files(file_list, target_folder):
    for f in tqdm(file_list):
        shutil.copy(f, target_folder)

# ----------------------------
# MAIN PIPELINE
# ----------------------------
def create_exp2_dataset():
    make_dirs()

    normal_imgs, abnormal_imgs = load_images()

    # Split Normal
    n_train, n_val, n_test = split_data(normal_imgs)
    # Split Abnormal
    a_train, a_val, a_test = split_data(abnormal_imgs)

    # Copy Normal
    copy_files(n_train, os.path.join(OUTPUT_DIR, "train/Normal"))
    copy_files(n_val, os.path.join(OUTPUT_DIR, "val/Normal"))
    copy_files(n_test, os.path.join(OUTPUT_DIR, "test/Normal"))

    # Copy Abnormal
    copy_files(a_train, os.path.join(OUTPUT_DIR, "train/Abnormal"))
    copy_files(a_val, os.path.join(OUTPUT_DIR, "val/Abnormal"))
    copy_files(a_test, os.path.join(OUTPUT_DIR, "test/Abnormal"))

    print("\n✔ EXP2 dataset created successfully!")
    print("Path:", OUTPUT_DIR)


if __name__ == "__main__":
    create_exp2_dataset()
