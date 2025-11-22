




# FERPlus-CNN: Facial Emotion Classification Web App

## Overview
This project implements a **Facial Emotion Recognition (FER)** web application using a **CNN-based model** trained on the **FERPlus dataset**. Users can upload an image, and the app predicts the facial emotion present in the image.

**Supported Emotions:**  
- Angry  
- Disgust  
- Fear  
- Happy  
- Sad  
- Surprise  
- Neutral  

The app is built with **Flask** for the backend and **TensorFlow/Keras** for the model.

---

## Dataset
The model is trained on the **FERPlus dataset**:
- Grayscale images
- Size: 112x112
- Seven emotion categories (listed above)

---

## Model Architecture
- Convolutional Neural Network (CNN)  
- Multiple convolution and pooling layers  
- Batch normalization after each convolution  
- Fully connected dense layers before output  
- Output layer: Softmax with 7 classes  

**Input shape:** `(112, 112, 1)`  
**Loss function:** `categorical_crossentropy`  
**Optimizer:** `Adam`  

---

## Features
- Upload an image via web interface
- Detect faces using OpenCV Haar Cascade
- Predict emotion using trained CNN
- Preview uploaded image with auto-crop for long/tall images
- Modern, responsive, clean web UI

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd <repo-folder>
