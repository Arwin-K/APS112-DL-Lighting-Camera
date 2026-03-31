# APS112 DL Lighting Camera

A deep learning project exploring advanced techniques for optimizing lighting and camera parameters using neural networks.

## Overview

This repository contains implementations and experiments our design prototype, focusing on deep learning applications in imaging science. The project investigates how machine learning models can be used to enhance and optimize lighting conditions and camera settings for improved image quality.

## Features

- **Deep Learning Models**: Neural network architectures for lighting and camera optimization
- **Jupyter Notebooks**: Interactive tutorials and experimental demonstrations
- **Python Implementation**: Core algorithms and utilities for model training and inference
- **Data Processing**: Tools for preparing and augmenting imaging datasets

## Project Structure

```
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── src/               # Core Python modules and utilities
├── models/            # Pre-trained and trained model weights
├── data/              # Dataset storage and preprocessing
├── results/           # Experimental results and outputs
└── README.md          # This file
```

## Requirements

- Python 3.8+
- TensorFlow or PyTorch (depending on implementation)
- NumPy
- Matplotlib
- Jupyter Notebook

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Arwin-K/APS112-DL-Lighting-Camera.git
cd APS112-DL-Lighting-Camera
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install Jupyter for running notebooks:
```bash
pip install jupyter
```

## Usage

### Running Experiments

To run the Jupyter notebooks:
```bash
jupyter notebook
```

Navigate to the `notebooks/` directory and open any `.ipynb` file to explore the experiments.

### Training Models

```python
python src/train.py --config configs/model_config.yaml
```

### Making Predictions

```python
python src/inference.py --model <path_to_model> --input <image_path>
```

## Project Goals

- Investigate the relationship between lighting parameters and image quality
- Develop deep learning models for automated camera calibration
- Optimize neural network architectures for real-time inference
- Create a reproducible framework for imaging optimization


*Last Updated: March 31, 2026*
