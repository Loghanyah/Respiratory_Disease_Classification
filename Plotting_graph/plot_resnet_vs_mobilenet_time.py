import matplotlib.pyplot as plt
import numpy as np

models = ["ResNet-50", "MobileNetV2"]
inference_time = [0.0182, 0.0216]

plt.figure()
plt.bar(models, inference_time)
plt.ylabel("Inference Time (seconds)")
plt.title("Inference Time Comparison (224×224)")
plt.tight_layout()
plt.show()