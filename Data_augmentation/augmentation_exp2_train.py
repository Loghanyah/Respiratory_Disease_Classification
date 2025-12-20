from PIL import Image
import torchvision.transforms as T
import os

TRAIN_DIR = "/Volumes/rloghanyah/PSM/Augmented_Dataset_Split_New/PNG/Exp2-Normal_vs_Abnormal/train"

# ---------------------------
# 1. Count images per class
# ---------------------------
class_sizes = {}
for cls in os.listdir(TRAIN_DIR):
    class_dir = os.path.join(TRAIN_DIR, cls)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(".png")]
    class_sizes[cls] = len(images)

print("Class sizes before augmentation:", class_sizes)

# Set target count = size of largest class
target_count = max(class_sizes.values())
print(f"Target count per class = {target_count}")

# ---------------------------
# 2. Augmentation pipeline
# ---------------------------
augment = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(12),
    T.ColorJitter(brightness=0.25, contrast=0.25),
    T.RandomResizedCrop(224, scale=(0.75, 1.0)),
])

# ---------------------------
# 3. Augment each class
# ---------------------------
for cls in os.listdir(TRAIN_DIR):
    class_dir = os.path.join(TRAIN_DIR, cls)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(".png")]

    count = len(images)
    idx = 0

    while count < target_count:
        img_path = os.path.join(class_dir, images[idx % len(images)])
        img = Image.open(img_path).convert("RGB")

        aug_img = augment(img)
        save_path = os.path.join(class_dir, f"aug_{count}.png")
        aug_img.save(save_path)

        count += 1
        idx += 1

    print(f"[DONE] {cls}: {count} images")

print("Augmentation complete!")