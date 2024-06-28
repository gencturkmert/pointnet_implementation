import os
from pickle import FALSE
import torch
import numpy as np
import zipfile
import urllib.request
from torch.serialization import normalize_storage_type
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class ModelNetDataset(Dataset):
    def __init__(self, root_dir, num_points=1024, download=False, augment=False, split='train', split_ratio=0.8, normalize=True,dataset='ModelNet10'):
        self.root_dir = root_dir
        self.num_points = num_points
        self.augment = augment
        self.split = split
        self.filepaths = []
        self.dataset = dataset
        
        if dataset == 'ModelNet40':
            self.url = "http://modelnet.cs.princeton.edu/ModelNet40.zip"
        elif dataset == 'ModelNet10':
            self.url = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
        else:
            raise ValueError("Invalid dataset. Choose either 'ModelNet10' or 'ModelNet40'.")

        dataset_path = os.path.join(self.root_dir, self.dataset)
        if not os.path.exists(dataset_path):
            print(f"Dataset {self.dataset} not found in {self.root_dir}.")
            download = True

        if download:
            self.download()

        # Load all filepaths initially
        temp_filepaths = []
        categories = [folder for folder in os.listdir(os.path.join(self.root_dir, self.dataset)) if os.path.isdir(os.path.join(self.root_dir, "ModelNet10", folder))]
        self.categories = categories  # Keep track of categories
        for category in categories:
            if split in ['train', 'validate']:
                folder = os.path.join(self.root_dir, self.dataset, category, 'train')
            elif split == 'test':
                folder = os.path.join(self.root_dir, self.dataset, category, 'test')
            files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.off')]
            temp_filepaths.extend([(file, categories.index(category)) for file in files])  # Store file paths and their corresponding labels

        if split == 'train':
            self.filepaths, _ = train_test_split(temp_filepaths, train_size=split_ratio, test_size=1-split_ratio, random_state=42)
        elif split == 'validate':
            _, self.filepaths = train_test_split(temp_filepaths, train_size=split_ratio, test_size=1-split_ratio, random_state=42)
        else:  # split == 'test'
            self.filepaths = temp_filepaths

    def download(self):
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        zip_path = os.path.join(self.root_dir, f'{self.dataset}.zip')
        if not os.path.exists(zip_path):
            print(f"Downloading {self.dataset} dataset...")
            urllib.request.urlretrieve(self.url, zip_path)
            print("Download complete.")
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root_dir)
        print("Extraction complete.")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath, label = self.filepaths[idx]
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} does not exist")
        points = self.load_mesh(filepath)
        points = self.sample_points(points)
        if self.augment:
            points = self.augment_points(points)
        point_cloud = torch.tensor(points, dtype=torch.float)
        return self.normalize_point_cloud(point_cloud), label

def load_mesh(self, filepath):
    vertices = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        header = lines[0].strip()
        if header.startswith('OFF'):
            if header == 'OFF':
                parts = lines[1].strip().split()
            else:
                parts = header[3:].strip().split()
        else:
            raise ValueError(f'Not a valid OFF file {filepath}')

        num_vertices = int(parts[0]) 
        num_faces = int(parts[1])    
        for line in lines[2:2 + num_vertices]:
            parts = line.strip().split()
            vertices.append([float(part) for part in parts])

    return np.array(vertices)


    def sample_points(self, points):
        if len(points) == 0:
            raise ValueError("No points found in the mesh")
        if len(points) >= self.num_points:
            idx = np.random.choice(len(points), self.num_points, replace=False)
        else:
            idx = np.random.choice(len(points), self.num_points, replace=True)
        return points[idx]

    def normalize_point_cloud(self,point_cloud):
        centroid = torch.mean(point_cloud, dim=0)
        point_cloud = point_cloud - centroid
        max_distance = torch.max(torch.sqrt(torch.sum(point_cloud ** 2, dim=1)))
        point_cloud = point_cloud / max_distance
        return point_cloud

    def augment_points(self, points):
        angle = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        points_rotated = np.dot(points, rotation_matrix)
        return points_rotated
