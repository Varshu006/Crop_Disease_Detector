# 🍅 Tomato Crop Disease Detector with Smart Advisory System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Varshu006/Crop_Disease_Detector/blob/main/another_copy_of_crop_detector.py)

An end-to-end Computer Vision project leveraging Deep Learning to identify 9 different types of tomato plant diseases and provide real-time actionable agricultural advice to farmers.

---

## 🚀 Project Overview
In agriculture, early and accurate detection of crop diseases is vital for maximizing yield and minimizing financial loss. This project builds an automated image classification pipeline that analyzes leaf photos, classifies the infection with high precision, and prints tailored management guidelines (such as specific fungicide or organic treatment advice).

- **Core Approach:** Deep Learning (CNN) using Transfer Learning
- **Frameworks:** TensorFlow / Keras & Python
- **Key Outcome:** Real-time diagnosis across **10 distinct classes** (9 diseases + 1 healthy state) accompanied by an instant mitigation advisory.

---

## 🛠️ Technical Stack & Frameworks
- **Deep Learning Framework:** TensorFlow 2.x, Keras
- **Pre-trained Architecture:** MobileNetV2 (Weights pre-trained on ImageNet)
- **Data Pipeline & Augmentation:** ImageDataGenerator (Keras)
- **Scientific Computing:** NumPy
- **Development Environment:** Google Colab

---

## 📊 Deep Learning Pipeline & Model Architecture

### 1. Data Pipeline & Augmentation
To optimize the model's robustness and protect against overfitting on limited agricultural datasets, raw plant images were systematically preprocessed:
- **Rescaling:** Pixel values normalized from `[0, 255]` to `[0, 1]`.
- **Validation Split:** 20% of data held out exclusively for cross-validation testing.
- **Augmentation Techniques:** Applied dynamic structural transformations including a `20-degree rotation range` and `horizontal flips` to simulate varying outdoor field lighting and camera angles.
- **Target Dimensionality:** Images resized uniformly to $(224 \times 224 \times 3)$.

### 2. Neural Network Architecture
The project utilizes **Transfer Learning** via a streamlined, mobile-optimized feature extractor to keep deployment light and efficient:
* **Base Model:** `MobileNetV2` (with top layers excluded to retain general feature maps). Base layers were explicitly frozen (`trainable = False`) to prevent gradient destruction during the initial epochs.
* **Global Average Pooling 2D:** Reduces spatial dimensions smoothly while retaining semantic features.
* **Dense Layer:** 128 hidden neurons utilizing `ReLU` activation for non-linear transformations.
* **Regularization (Dropout):** Implemented a `0.3` dropout layer to actively prevent overfitting.
* **Output Classification Layer:** Multi-class Softmax layer maps directly to the active tomato condition classes.

```text
Input (224x224x3) ➔ MobileNetV2 Base ➔ Global Average Pooling ➔ Dense (128, ReLU) ➔ Dropout (0.3) ➔ Output (Softmax)
