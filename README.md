# false_positive_elimination

## Introduction

Welcome to the false_positive_elimination project! This repository hosts a machine learning application aimed at classifying false positives using a classification algorithm built on the timm library and utilizing PyTorch Lightning. This project is particularly useful for improving the accuracy of object detection models by effectively distinguishing false positives.

## Project Structure

```bash
false_positive_classification/
├── docker/
│   └── Dockerfile
├── LICENSE
├── README.md
├── classifier.py
├── model.py
├── get_optim.py
├── inference.py
├── convert_tensorrt.py
├── convert_onnx.py
├── train.py
├── train.yaml
├── data/
│   ├── augmentations.py
│   ├── dataset.py
└── datasets

```

## Getting Started

### Prerequisites
Before you begin, ensure you have Docker installed on your machine. If you don't have Docker installed, you can download it from [Docker's official website](https://docs.docker.com/engine/install/)

### Installation with Docker

#### Clone the Repository:
```
git clone https://github.com/taylanates24/false_positive_classification.git
cd false_positive_classification
```

#### Build the Docker Image:
Build an image from a Dockerfile in the project directory. This will install all necessary dependencies.
```
docker build -t false_positive_classification -f docker/Dockerfile .
```
#### Run the container
Start a container from the image you've just built.
```
docker run -v $(pwd):/workspace -it --gpus all --rm --ipc host false_positive_classification:latest
```
