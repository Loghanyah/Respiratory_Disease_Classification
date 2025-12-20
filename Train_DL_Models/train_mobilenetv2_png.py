import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.optim as optim
import time
import csv
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# ----------------------------
# CONFIG - EDIT ONLY THESE
# ----------------------------
DATA_DIR = "/Volumes/rloghanyah/PSM/Dataset_Split_New/PNG/Exp1-6_class" # Original: Dataset_Split_New & Augmented: Augmented_Dataset_Split_New
BATCH_SIZE = 32
NUM_CLASSES = 2
LR = 0.0001
EPOCHS = 20

# Change IMG_SIZE to 64 / 128 / 224 before running
IMG_SIZE = 224
EXP_NAME = f"MobilenetV2_Exp_ImageSize_{IMG_SIZE}"

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
#print("Using device:", DEVICE)

# ----------------------------
# CSV BENCHMARK SETUP
# ----------------------------
CSV_LOG_FILE = f"benchmark_{EXP_NAME}.csv"
if not os.path.exists(CSV_LOG_FILE):
    with open(CSV_LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Epoch", "Train_Loss", "Train_Acc",
            "Val_Acc", "Val_Precision", "Val_Recall", "Val_F1",
            "Epoch_Time(sec)"
        ])

# ----------------------------
# TRANSFORMS (ON-THE-FLY)
# ----------------------------
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ----------------------------
# DATASETS & LOADERS
# ----------------------------
train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
val_data = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_test_transforms)
test_data = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_test_transforms)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ----------------------------
# MODEL: MobileNetV2 (lightweight)
# ----------------------------
# Use pretrained weights when available. If torchvision version warns, fallback to pretrained=True.
try:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
except Exception:
    # fallback for older torchvision
    model = models.mobilenet_v2(pretrained=True)

# replace classifier
if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, NUM_CLASSES)
else:
    # generic fallback
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.last_channel, NUM_CLASSES))

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ----------------------------
# TRAINING LOOP
# ----------------------------
def train_model():
    best_acc = 0.0
    train_losses, train_accs, val_accs = [], [], []

    for epoch in range(EPOCHS):
        start_time = time.time()
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print("-" * 30)

        # train
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            running_corrects += torch.sum(preds == labels)

        epoch_loss = running_loss / len(train_data)
        epoch_acc = running_corrects.float() / len(train_data)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())

        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

        # val
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)

        val_acc = np.mean(val_preds == val_labels)
        val_precision = precision_score(val_labels, val_preds, average="weighted", zero_division=0)
        val_recall = recall_score(val_labels, val_preds, average="weighted", zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, average="weighted", zero_division=0)

        val_accs.append(val_acc)
        epoch_time = time.time() - start_time

        # CSV logging
        with open(CSV_LOG_FILE, mode="a", newline="") as f:
            csv.writer(f).writerow([
                epoch+1, round(epoch_loss,5), round(epoch_acc.item(),5),
                round(val_acc,5), round(val_precision,5), round(val_recall,5), round(val_f1,5),
                round(epoch_time,2)
            ])

        print(f"Val Acc: {val_acc:.4f} | Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f}")
        print(f"Epoch Time: {epoch_time:.2f} sec\n")

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"mobilenetv2_{EXP_NAME}.pth")
            print("✔ Best model updated\n")

    # plot curves
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(range(1, len(train_accs)+1), train_accs, label="Train Acc")
    plt.plot(range(1, len(val_accs)+1), val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{EXP_NAME}_training_validation_curves.png")
    plt.show()
    plt.close()
    print(f"Training complete. Best Val Acc: {best_acc}")

# ----------------------------
# EVALUATE TEST
# ----------------------------
def evaluate_test():
    checkpoint = f"mobilenetv2_{EXP_NAME}.pth"
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    all_preds, all_labels = [], []
    inference_times = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            start = time.time()
            outputs = model(imgs)
            end = time.time()

            _, preds = torch.max(outputs, 1)
            inference_times.append(end - start)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    avg_time = sum(inference_times) / len(inference_times)
    fps = 1 / avg_time if avg_time>0 else float('inf')

    print("Test Accuracy:", test_acc)
    print(f"Avg Inference Time per Batch: {avg_time:.6f} sec  | FPS (per batch): {fps:.2f}")

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_data.classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Test Confusion Matrix - {EXP_NAME}")
    plt.tight_layout()
    plt.show()

    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=train_data.classes))

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    print("Using device:", DEVICE)
    train_model()
    evaluate_test()