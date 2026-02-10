# Respiratory Disease Classification Using Deep Learning

This repository contains the full implementation of the Final Year Project / PSM titled:

**“Respiratory Disease Classification from Lung Sound Signals Using Deep Learning Techniques”**

The codebase directly supports the methodology, experiments, and results presented in the accompanying thesis report. It is intended for **academic verification, panel evaluation, and future research extension**.

GitHub Repository:  
https://github.com/Loghanyah/Respiratory_Disease_Classification

---

# Setup

``` 
pip install -r requirements.txt
```

## 1. Project Motivation

Respiratory disease diagnosis using lung auscultation is widely practiced due to its non-invasive and low-cost nature. However, traditional auscultation is subjective and highly dependent on clinician experience, particularly when differentiating diseases with overlapping acoustic characteristics such as asthma, COPD, pneumonia, and heart failure.

This project proposes an **automated, deep learning-based respiratory disease classification framework** using biologically inspired **gammatonegram time–frequency representations** and convolutional neural networks (CNNs). Emphasis is placed not only on classification accuracy, but also on **computational efficiency, inference time, and deployability**.

---

## 2. Scope of This Repository

This repository includes:

- Lung sound preprocessing pipeline
- Gammatonegram generation using auditory-inspired filterbanks
- Multi-class classification for six respiratory conditions:
  - Asthma
  - Bronchiectasis
  - COPD
  - Heart Failure
  - Pneumonia
  - Normal
- Comparative evaluation of:
  - **ResNet-50 (heavyweight model)**
  - **MobileNetV2 (lightweight model)**
- Image resolution analysis:
  - 64 × 64
  - 128 × 128
  - 224 × 224
- Training with and without data augmentation
- Comprehensive benchmarking:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix
  - Inference time

All experimental results reported in the thesis were generated using this codebase.

---

## 3. Dataset Notes

⚠️ **Clinical Data Disclaimer**

Due to ethical and privacy constraints, raw lung sound recordings are **not included** in this repository.

The dataset used in this study was:
- Provided by hospital researchers
- Recorded using a digital stethoscope
- Preprocessed and organized into six disease classes

Users wishing to reproduce the experiments must supply their own lung sound recordings and follow the same preprocessing and organization described in **Chapter 3 (Methodology)** of the thesis.

---


## 4. Repository Structure

---

## 5. Preprocessing and Feature Extraction

Lung sound recordings undergo:
1. Noise filtering
2. Automated segmentation
3. Amplitude normalization

Gammatonegrams are generated using a **gammatone filterbank**, which models the frequency selectivity of the human cochlea. This representation was selected based on prior comparative studies demonstrating superior performance over spectrograms, scalograms, and mel-spectrograms for lung sound classification.

Details are provided in **Sections 3.4 and 3.5** of the thesis.

---

## 6. Model Training

Two CNN architectures are evaluated:

### Heavyweight Model
- **ResNet-50**
- High representational capacity
- Strong performance at higher resolutions
- Higher computational cost

### Lightweight Model
- **MobileNetV2**
- Depthwise separable convolutions
- Reduced parameter count
- Faster inference time
- Suitable for resource-constrained environments

Training is performed using:
- 70% Training
- 15% Validation
- 15% Testing

Data augmentation is applied **only to the training set** to address class imbalance.

---

## 7. Experimental Design

Experiments evaluate:
- Effect of image resolution
- Impact of data augmentation
- Trade-off between accuracy and efficiency
- Class-wise performance consistency
- Inference time suitability for real-world deployment

This aligns directly with **Chapters 4 and 5** of the thesis.

---

## 8. Intended Use

This repository is intended for:
- Thesis verification by academic panels
- Reproducibility and transparency
- Future research on respiratory sound analysis
- Extension to edge or real-time diagnostic systems

It is **not** intended for direct clinical deployment without further validation.

---

## 9. Ethical and Academic Notice

- No patient-identifiable data is included
- Users must obtain appropriate ethical clearance for clinical data
- Results should not be used for medical diagnosis without clinical approval

---

## 10. Citation

If you use or adapt this work, please cite the corresponding thesis and relevant referenced publications listed in the report.

---
