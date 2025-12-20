import matplotlib.pyplot as plt

image_sizes = [64, 128, 224]
inference_time = [0.0148, 0.0151, 0.0216]  # your real results

plt.figure()
plt.plot(image_sizes, inference_time, marker='o')
plt.xlabel("Image Size (pixels)")
plt.ylabel("Inference Time per Batch (sec)")
plt.title("Image Size vs Inference Time")
plt.grid(True)
plt.savefig("inference_time_vs_image_size(MobilenetV2).png")
plt.show()