import os
import shutil
import random

# === Settings ===
input_root = "/Volumes/rloghanyah/PSM/Gammatonegram/PNG/Augmented_images_1k"
output_root = "/Volumes/rloghanyah/PSM/Dataset_Split/PNG/Exp1-6_class"

train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# Ensure reproducibility
random.seed(42)

# Create output folders
splits = ['train', 'val', 'test']
classes = os.listdir(input_root)

for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(output_root, split, cls), exist_ok=True)

# Process each class
for cls in classes:
    class_folder = os.path.join(input_root, cls)
    files = [
        f for f in os.listdir(class_folder)
        if f.lower().endswith(".png") and not f.startswith("._")
    ]

    random.shuffle(files)

    total = len(files)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    # Copy to new folders
    for f in train_files:
        shutil.copy(
            os.path.join(class_folder, f), 
            os.path.join(output_root, "train", cls, f)
        )

    for f in val_files:
        shutil.copy(
            os.path.join(class_folder, f), 
            os.path.join(output_root, "val", cls, f)
        )

    for f in test_files:
        shutil.copy(
            os.path.join(class_folder, f), 
            os.path.join(output_root, "test", cls, f)
        )
    
    print(f"Found {len(files)} PNG files in class {cls}")

    print(f"Class {cls}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

print("✅ Dataset splitting completed!")