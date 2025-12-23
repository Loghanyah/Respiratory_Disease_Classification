import matplotlib.pyplot as plt
import numpy as np

models = ["ResNet-50", "MobileNetV2"]
test_accuracy = [0.7843, 0.8076]
f1_score = [0.78, 0.81]

x = np.arange(len(models))
width = 0.35

plt.figure()
plt.bar(x - width/2, test_accuracy, width, label="Test Accuracy")
plt.bar(x + width/2, f1_score, width, label="Weighted F1-score")

plt.xticks(x, models)
plt.ylabel("Score")
plt.title("Comparison of Test Accuracy and F1-score (224×224)")
plt.legend()
plt.tight_layout()
plt.show()