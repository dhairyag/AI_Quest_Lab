# AI_Quest_Lab

A collection of AI-ML projects showcasing different applications and technologies.
Each project folder contains detailed documentation, setup instructions, and technical details in their respective README files.

## Projects

### 1. [Markdown Notes Chrome Extension](session_01_chrome_extension/README.md)
A Chrome extension built to capture and format web content in Markdown. Features text selection, formatting options, and note organization capabilities. Built with JavaScript and Chrome Extension APIs.

### 2. [Image and File Processor](session_02_web_FastAPI_app/README.md)
A web application for processing images and files, built with FastAPI backend and HTML/CSS/JavaScript frontend. Provides a clean interface for file uploads and processing operations.

### 3. [DataAlchemy: Data Processing App](session_03_ML_data_processing_app/README.md)
A comprehensive FastAPI-based application for processing multiple data types (text, audio, image, 3D geometry). Features preprocessing, augmentation, and visualization capabilities. Built with FastAPI, PyTorch, and various data processing libraries.

### 4. [WebApp MNIST Monitor](session_04_web_app_mnist_monitor/README.md)
A real-time web-based monitoring system for training Convolutional Neural Networks (CNNs) on the MNIST dataset. This system provides live updates on training progress, loss curves, and sample predictions through an intuitive web interface.

### 5. [MNIST CI/CD Pipeline](https://github.com/dhairyag/MINIST_CICD) (repo)
A lightweight Convolutional Neural Network (CNN) implementation for MNIST digit classification using PyTorch.
The project includes a robust CI/CD pipeline implemented with GitHub Actions. All changes must pass the automated pipeline before deployment. 

### 6. [Tiny-MNIST with GitHub Actions](https://github.com/dhairyag/tiny_MNIST) (repo)
This repository implements a CNN-based deep learning model for MNIST digit classification with automated architecture validation through GitHub Actions.
`main_mnist.py` uses only 3,130 parameters to achieve more than 99.4% accuracy, observed in multiple runs.

### 7. [MNIST Classification Experiments on SageMaker](https://github.com/dhairyag/multi_models_MNIST) (repo)
This repository contains experiments for classifying the MNIST dataset using different neural network architectures. The experiments are conducted in three Jupyter notebooks: `01_train_mnist.ipynb`, `02_train_mnist.ipynb`, and `03_train_mnist.ipynb`. Each notebook explores a different model architecture and training strategy.

### 8. [CIFAR10 Image Classification with Custom CNN](https://github.com/dhairyag/4blocks_CIFAR10) (repo)
This project implements a custom Convolutional Neural Network (CNN) architecture for the CIFAR10 dataset classification task. The network achieves 85%+ accuracy while maintaining under 128k parameters through efficient architecture choices and modern convolution techniques.

### 9. [ImageNet1k Classification with ResNet50](https://github.com/dhairyag/ImageNet1k_ResNet50) (repo)
PyTorch implementation of ResNet50 training on ImageNet1000, with a focus on cloud deployment using AWS EC2. The project achieves 70% accuracy after 64 epochs using a `g4dn.2xlarge` instance with a NVIDIA T4 GPU. It includes comprehensive features like distributed training, mixed precision, and various optimizations, along with detailed performance monitoring and visualization of training metrics.

### 10. [NanoGPT Training Implementation](https://github.com/dhairyag/ShakespeareGPT-Forge) (repo)
A PyTorch implementation of GPT-2 style transformer with modern training optimizations. This implementation follows the architecture described in the GPT-2 paper while incorporating various performance enhancements for efficient training.

### 11. [SmolLM2: Implementation from the ground up](https://github.com/dhairyag/SmolLM2_GroundUp) (repo)
This repository contains a PyTorch implementation of the SmolLM2 language model, a compact transformer-based model with grouped-query attention, designed for efficient natural language processing. Built from the ground up through reverse engineering, this implementation closely follows the architecture described in the model's Hugging Face specifications, primarily based on the model's Hugging Face model card (hf_model.md) and configuration files (config_smollm2_135M.yaml).

### 12. [LLM Docker Microservices](https://github.com/dhairyag/docker_LLM_microservices) (repo)
A containerized deployment of SmolLM2 language model using a microservices architecture. The project demonstrates Docker container communication with one container serving the model and another handling user interactions.
