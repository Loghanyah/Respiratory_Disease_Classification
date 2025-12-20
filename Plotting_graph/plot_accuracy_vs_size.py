import matplotlib.pyplot as plt

# Your real results
image_sizes = [64, 128, 224]
accuracies = [0.72, 0.73, 0.81]   # ← replace if needed

plt.figure()
plt.plot(image_sizes, accuracies, marker='o')
plt.xlabel("Image Size (pixels)")
plt.ylabel("Test Accuracy")
plt.title("Image Size vs Classification Accuracy")
plt.grid(True)
plt.savefig("accuracy_vs_image_size(MobilenetV2).png")
plt.show()