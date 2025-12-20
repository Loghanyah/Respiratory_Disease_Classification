import matplotlib.pyplot as plt

# Image Sizes
image_sizes = [64, 128, 224]

# ResNet-50 Epoch Time (seconds) — CHANGE THESE
resnet_time = [43, 102, 249]   # <<< PUT YOUR REAL VALUES

# MobileNetV2 Epoch Time (seconds) — CHANGE THESE
mobilenet_time = [24, 48, 115]  # <<< PUT YOUR REAL VALUES

plt.figure()
plt.plot(image_sizes, resnet_time, marker='o', label="ResNet-50")
plt.plot(image_sizes, mobilenet_time, marker='o', label="MobileNetV2")

plt.xlabel("Image Size (pixels)")
plt.ylabel("Epoch Training Time (seconds)")
plt.title("Training Time Comparison: ResNet-50 vs MobileNetV2")
plt.legend()
plt.grid(True)

plt.savefig("resnet_vs_mobilenet_training_time.png")
plt.show()