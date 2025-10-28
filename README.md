# Image Classifier with Python

A deep learning-based image classification project using Convolutional Neural Networks (CNN) and the CIFAR-10 dataset, featuring an intuitive Tkinter GUI for real-time image classification.

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)

## Overview

This project implements a CNN-based image classifier that can recognize and classify images into 10 different categories from the CIFAR-10 dataset. The application provides a user-friendly graphical interface for uploading images and receiving instant classification results.

## Features

- **Deep Learning Model**: Custom CNN architecture with multiple convolutional and dense layers
- **CIFAR-10 Dataset**: Trained on 60,000 images (50,000 training, 10,000 testing)
- **Interactive GUI**: Built with Tkinter for easy image upload and classification
- **Real-time Classification**: Instant prediction results displayed in the interface
- **Model Persistence**: Trained model saved as HDF5 file for reuse
- **Data Preprocessing**: Automatic image normalization and resizing

## Classified Categories

The model can classify images into the following 10 classes:

- ✈️ Aeroplane
- 🚗 Automobile
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐕 Dog
- 🐸 Frog
- 🐴 Horse
- 🚢 Ship
- 🚚 Truck

## Model Architecture

The CNN model consists of:

1. **Convolutional Layers**:
   - Conv2D (32 filters, 3x3 kernel) with ReLU activation
   - Dropout (0.2)
   - Conv2D (32 filters, 3x3 kernel) with ReLU activation
   - MaxPooling2D (2x2 pool size)

2. **Fully Connected Layers**:
   - Flatten layer
   - Dense (64 neurons, ReLU)
   - Dense (512 neurons, ReLU)
   - Dropout (0.5)
   - Dense (10 neurons, Softmax) - Output layer

3. **Optimization**:
   - Optimizer: SGD (learning rate: 0.01, momentum: 0.9)
   - Loss function: Categorical Crossentropy
   - Metrics: Accuracy
   - Training: 10 epochs with batch size of 32

## Requirements

```
tensorflow>=2.0.0
keras>=2.0.0
opencv-python
matplotlib
pillow
numpy
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Image-Classifier-with-Python.git
cd Image-Classifier-with-Python
```

2. Install required dependencies:
```bash
pip install tensorflow keras opencv-python matplotlib pillow numpy
```

## Usage

### Training the Model

Run the main script to train the model:
```bash
python "Image Classifier Project.py"
```

The script will:
1. Load and preprocess the CIFAR-10 dataset
2. Build the CNN model architecture
3. Train the model for 10 epochs
4. Display training accuracy
5. Save the trained model as `model1_cifar_10epoch.h5`
6. Launch the GUI application

### Using the GUI Application

1. **Upload Image**: Click the "Upload Image" button to select an image file
2. **Classify**: Click the "Classify the Image" button to get the prediction
3. **View Results**: The classification result will be displayed in the interface
4. **Data Information**: Click "Get Data Information" for dataset details

## Project Structure

```
Image-Classifier-with-Python/
│
├── Image Classifier Project.py    # Main application file
├── README.md                       # Project documentation
└── model1_cifar_10epoch.h5        # Trained model (generated after training)
```

## Performance

The model achieves competitive accuracy on the CIFAR-10 test dataset. The exact accuracy is displayed in the console after training completion.

## Technical Details

- **Input Shape**: 32x32x3 (RGB images)
- **Normalization**: Pixel values scaled to [0, 1]
- **Data Augmentation**: None (can be added for improved performance)
- **Kernel Constraint**: MaxNorm(3) for regularization
- **Dropout**: Used to prevent overfitting

## Author

**Enes Günümdoğdu**  

