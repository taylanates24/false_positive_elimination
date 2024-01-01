# false_positive_elimination

## Introduction

Welcome to the False Fositive Elimination project! This repository hosts a machine learning application aimed at classifying false positives using a classification algorithm built on the timm library and utilizing PyTorch Lightning. This project is particularly useful for improving the accuracy of object detection models by effectively distinguishing false positives.

## Project Structure

```bash
false_positive_elimination/
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
docker build -t false_positive_elimination -f docker/Dockerfile .
```
#### Run the container
Start a container from the image you've just built.
```
docker run -v $(pwd):/workspace -it --gpus all --rm --ipc host false_positive_elimination:latest
```
#### Training
1. Change the `train.yaml` file with appropriate variables.
    - Choose the number of epochs, learning_rate, validation frequency, optimizer, learning rate scheduler, number of GPUs, batch size, augmentations and their parameters.
2. Run the following code to train the code.
```
python3 train.py --train_cfg train.yaml
```
#### Choosing The Right Confidence Threshold
**Coming Soon!**
The model eventually gives a probability (a softmax output.) We will draw a Precision-Recall curve to choose the right confidence threshold so that you can have a Precision of 1. That means you do not eliminate any true positives!

#### About the Model

In this project, I use the `timm` library of the [Hugging Face](https://github.com/huggingface).
In this way, you can choose a model among a lot of pre-trained and deployed Hugging Face models by:
```
available_models = timm.list_models()
```
And you can use them by:
```
model = timm.create_model('tf_efficientnet_lite2', pretrained=pretrained, num_classes=num_classes)
```
I have chosen the [EfficientNet model](https://paperswithcode.com/paper/efficientnet-rethinking-model-scaling-for) because its accuracy and latency are perfect.
