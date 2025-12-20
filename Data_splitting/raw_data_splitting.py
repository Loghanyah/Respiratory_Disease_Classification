import os
import shutil
from sklearn.model_selection import train_test_split

RAW_DIR = "/Volumes/rloghanyah/PSM/Gammatonegram/PNG/Images"
OUTPUT_DIR = "/Volumes/rloghanyah/PSM/Dataset_Split_New/PNG/Exp1-6_class"

os.makedirs(OUTPUT_DIR, exist_ok=True)

splits = ['train', 'val', 'test']
classes = os.listdir(RAW_DIR)

for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# ------------------------
# SPLIT RATIOS
# ------------------------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

for cls in classes:
    class_path = os.path.join(RAW_DIR, cls)
    images = [f for f in os.listdir(class_path) if f.lower().endswith((".png"))]

    train_files, temp = train_test_split(images, train_size=(1-TRAIN_RATIO), random_state=42)
    val_files, test_files = train_test_split(temp, test_size=TEST_RATIO/(TEST_RATIO + VAL_RATIO), random_state=42)

    # Copy images to new folders
    for fname in train_files:
        shutil.copy(os.path.join(class_path, fname), os.path.join(OUTPUT_DIR, "train", cls))
    for fname in val_files:
        shutil.copy(os.path.join(class_path, fname), os.path.join(OUTPUT_DIR, "val", cls))
    for fname in test_files:
        shutil.copy(os.path.join(class_path, fname), os.path.join(OUTPUT_DIR, "test", cls))

print("Raw dataset has been split successfully.")