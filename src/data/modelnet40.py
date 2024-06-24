import os
import torch
import numpy as np
import zipfile
import urllib.request
from torch.utils.data import Dataset

class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir, categories=None, num_points=1024, download=True, augment=False):
        self.root_dir = root_dir
        self.categories = categories
        self.num_points = num_points
        self.augment = augment
        self.url = "http://modelnet.cs.princeton.edu/ModelNet40.zip"

        if download:
            self.download()

        self.filepaths = []
        if self.categories is None:
            self.categories = [folder for folder in os.listdir(os.path.join(self.root_dir, "ModelNet40")) if os.path.isdir(os.path.join(self.root_dir, "ModelNet40", folder))]

        for category in self.categories:
            folder = os.path.join(self.root_dir, "ModelNet40", category, 'train')
            files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.off')]
            self.filepaths.extend(files)

    def download(self):
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        zip_path = os.path.join(self.root_dir, 'ModelNet40.zip')
        if not os.path.exists(zip_path):
            print("Downloading ModelNet40 dataset...")
            urllib.request.urlretrieve(self.url, zip_path)
            print("Download complete.")
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root_dir)
        print("Extraction complete.")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        points = self.load_mesh(filepath)
        points = self.sample_points(points)
        if self.augment:
            points = self.augment_points(points)
        return torch.tensor(points, dtype=torch.float)

    def load_mesh(self, filepath):
        vertices = []
        with open(filepath, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith('vertex'):
                    parts = line.split()[1:]
                    vertices.append([float(part) for part in parts])
        return np.array(vertices)

    def sample_points(self, points):
        if len(points) >= self.num_points:
            idx = np.random.choice(len(points), self.num_points, replace=False)
        else:
            idx = np.random.choice(len(points), self.num_points, replace=True)
        return points[idx]

    def augment_points(self, points):
        # Rotate the point cloud about the z-axis
        angle = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        points_rotated = np.dot(points, rotation_matrix)
        return points_rotated

