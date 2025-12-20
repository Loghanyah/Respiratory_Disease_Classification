from PIL import Image
import torchvision.transforms as T
import os

AUGMENT_COUNT = 1000 # per class

TRAIN_DIR = "/Volumes/rloghanyah/PSM/Augmented_Dataset_Split_New/PNG/Exp2-Normal_vs_Abnormal/train"

augment = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
])

for cls in os.listdir(TRAIN_DIR):
    class_dir = os.path.join(TRAIN_DIR, cls)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith((".png"))]
    
    count = len(images)
    idx = 0

    while count < AUGMENT_COUNT:
        img_path = os.path.join(class_dir, images[idx % len(images)])
        img = Image.open(img_path)

        aug_img = augment(img)
        save_path = os.path.join(class_dir, f"aug_{count}.png")
        aug_img.save(save_path)

        count += 1
        idx += 1

    print(f"Completed augmentation for {cls}. Final count: {count}")