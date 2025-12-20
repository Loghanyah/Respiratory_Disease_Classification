import matplotlib.pyplot as plt

# Image Sizes
image_sizes = [64, 128, 224]

# ResNet-50 Results (YOUR REAL VALUES)
resnet_acc = [0.7201, 0.7405, 0.7843]

# MobileNetV2 Results (REPLACE with your real values)
mobilenet_acc = [0.7230, 0.7318, 0.8076]  # <<< CHANGE THESE 3

plt.figure()
plt.plot(image_sizes, resnet_acc, marker='o', label="ResNet-50")
plt.plot(image_sizes, mobilenet_acc, marker='o', label="MobileNetV2")

plt.xlabel("Image Size (pixels)")
plt.ylabel("Test Accuracy")
plt.title("ResNet-50 vs MobileNetV2 Accuracy Comparison")
plt.legend()
plt.grid(True)

plt.savefig("resnet_vs_mobilenet_accuracy.png")
plt.show()