
# PointNet Implementation

This repository contains a PyTorch implementation of [PointNet](https://arxiv.org/abs/1612.00593), a novel deep learning architecture designed to handle point clouds directly. The architecture was originally proposed by Charles R. Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas from Stanford University.

## Table of Contents
- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Environment Setup](#environment-setup)
- [How to Run](#how-to-run)
- [Datasets](#datasets)
- [References](#references)

## Introduction

PointNet is designed to classify and segment 3D point clouds. Unlike traditional methods that rely on converting point clouds to other representations (e.g., voxel grids), PointNet processes point sets directly. The network achieves permutation invariance over input points and uses max pooling to capture global information from the points.

This repository implements both the classification and segmentation modules as described in the original paper.

## Model Architecture

The core idea of PointNet is to apply a series of shared multi-layer perceptrons (MLPs) followed by max pooling to aggregate the global features of point clouds.

- **Input**: Unordered point sets
- **Output**: Classification or segmentation results
- **Modules**:
  1. Shared MLP layers for feature learning
  2. Max pooling layer for permutation invariance
  3. Fully connected layers for classification/segmentation

## Results

The following benchmarks are achieved on the ModelNet10 and ModelNet40 datasets:

- **ModelNet10 - Classification**: Accuracy: 82.93%
- **ModelNet40 - Classification**: Accuracy: 81.17%

## Environment Setup

To set up the environment and install the necessary dependencies, follow these steps:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/gencturkmert/pointnet_implementation.git
    cd pointnet_implementation
    ```

2. **Create a virtual environment** (optional but recommended):

    ```bash
    python3 -m venv pointnet_env
    source pointnet_env/bin/activate  # On Windows: pointnet_env\Scripts\activate
    ```

3. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Install PyTorch** (if not already installed):

    You can install PyTorch by visiting the [official PyTorch website](https://pytorch.org/get-started/locally/) and following the installation instructions based on your system configuration.

## How to Run

To train and evaluate the PointNet model:

1. **Training the model**:

    ```bash
    python train.py --dataset <dataset_path> --epochs <number_of_epochs>
    ```

2. **Testing the model**:

    ```bash
    python test.py --model <model_path> --dataset <dataset_path>
    ```

Replace `<dataset_path>` and `<model_path>` with the appropriate file paths.

## Datasets

PointNet is evaluated on the following datasets:

- **ModelNet10**
- **ModelNet40**

These datasets are 3D object datasets that provide point clouds for various objects, and they are commonly used to benchmark 3D deep learning models. You can download these datasets from the official ModelNet webpage.

## References

- Original PointNet paper: [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
- PyTorch: [PyTorch official site](https://pytorch.org/)

## License

This project is licensed under the MIT License.
