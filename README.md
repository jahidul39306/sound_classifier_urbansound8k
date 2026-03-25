# UrbanSound8K Sound Classifier

A deep learning-based sound classification system for urban environmental sounds using the UrbanSound8K dataset. This project implements a CNN-based architecture to classify audio samples into 10 urban sound categories.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview
This project aims to classify urban sounds using convolutional neural networks (CNNs). The system processes audio files, extracts mel-spectrogram features, and trains a deep learning model to identify various urban sound classes such as air conditioners, car horns, children playing, dog barks, drilling, engine idling, gunshots, jackhammers, sirens, and street music.

## ✨ Features
- **Audio Preprocessing**: Extracts mel-spectrograms from raw audio files
- **CNN Architecture**: Custom deep learning model optimized for sound classification
- **GPU Support**: Optimized for CUDA-enabled GPU training
- **Configurable Pipeline**: YAML-based configuration system for easy experimentation
- **Modular Codebase**: Clean, reusable components for data loading, model definition, and training
- **Visualization Tools**: Training metrics and spectrogram visualizations

## 📊 Dataset
The project uses the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html), which contains 8,732 labeled sound excerpts (≤4 seconds) from 10 urban sound classes:

| Class ID | Sound Class |
|----------|-------------|
| 0 | Air Conditioner |
| 1 | Car Horn |
| 2 | Children Playing |
| 3 | Dog Bark |
| 4 | Drilling |
| 5 | Engine Idling |
| 6 | Gun Shot |
| 7 | Jackhammer |
| 8 | Siren |
| 9 | Street Music |

## 📁 Project Structure
```text
sound_classifier_urbansound8k/
├── configs/                # Configuration files
│   └── config.yaml         # Main configuration for hyperparameters & paths
├── notebooks/              # Jupyter notebooks for EDA and experimentation
├── outputs/                # Training outputs
│   └── figures/            # Saved plots (Loss/Accuracy curves, Confusion Matrix)
├── src/                    # Core source code
│   ├── data/               # Scripts for data loading and MFCC preprocessing
│   ├── models/             # Neural network architecture definitions
│   ├── training/           # Training loops and evaluation logic
│   └── utils/              # Helper functions (logging, file handling)
├── main.py                 # Primary entry point to run the pipeline
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
