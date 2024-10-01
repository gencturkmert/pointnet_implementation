
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

To set up the environment and install the necessary dependencies using Conda, follow these steps:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/gencturkmert/pointnet_implementation.git
    cd pointnet_implementation
    ```

2. **Create and activate a Conda environment**:

    ```bash
    conda create --name pointnet_env python=3.9
    conda activate pointnet_env
    ```

3. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## How to Run

To train, validate, and test the PointNet model, you can use the provided `train.py` script. Here are the key arguments you can pass:

- `--num_points`: Number of points in each sample (default: 2048)
- `--num_classes`: Number of output classes (default: 40 for ModelNet40)
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of epochs to train (default: 100)
- `--dataset`: Dataset to use (`ModelNet10` or `ModelNet40`) (default: ModelNet10)
- `--dir_path`: Directory path for saving models and results (default: `/content/drive/MyDrive/pointnet_torch`)
- `--download`: Flag to download the dataset (1 to download, 0 to use existing)
- `--normalize`: Normalize the point cloud data (1 to normalize, 0 otherwise)

### Example Command

To train the model on ModelNet10 with 2048 points per sample and a batch size of 32 for 100 epochs:

```bash
python train.py --num_points 2048 --num_classes 10 --batch_size 32 --epochs 100 --dataset ModelNet10 --dir_path ./results --download 1 --normalize 1
```

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
