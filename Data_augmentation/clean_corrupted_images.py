from PIL import Image
import os

ROOT = "/Volumes/rloghanyah/PSM/Augmented_Dataset_Split_New/PNG/Exp2-Normal_vs_Abnormal"

def is_image_valid(path):
    try:
        img = Image.open(path)
        img.verify()  
        return True
    except:
        return False

def clean_folder(folder):
    removed = 0
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".png"):
                full = os.path.join(root, f)
                if not is_image_valid(full):
                    print("❌ Removing corrupted image:", full)
                    os.remove(full)
                    removed += 1
    return removed

print("Scanning for corrupted PNGs...")
removed = clean_folder(ROOT)
print(f"✔ Done. Removed {removed} corrupted images.")
