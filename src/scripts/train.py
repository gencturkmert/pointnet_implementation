import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import numpy as np
import time
import math

from data.modelnet10 import ModelNet10Dataset
from data.modelnet40 import ModelNet40Dataset
from model.pointnet import PointNet
from utils.plot import save_plot_loss_acc, save_confusion_matrix

def train_one_epoch(model, train_loader, device, optimizer, criterion, scaler):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total = 0
    
    # Create a progress bar
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for data, labels in progress_bar:
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs, _, _ = model(data)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item()

        # Update the progress bar description
        progress_bar.set_postfix(loss=running_loss/len(train_loader), accuracy=total_correct/total*100)

    train_loss = running_loss / len(train_loader)
    train_accuracy = total_correct / total * 100
    return train_loss, train_accuracy

def validate(model, val_loader, device, criterion):
    model.eval()
    total_correct = 0
    total = 0
    running_loss = 0.0
    
    # Create a progress bar
    progress_bar = tqdm(val_loader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for data, labels in progress_bar:
            data, labels = data.to(device), labels.to(device)
            with autocast():
                outputs, _, _ = model(data)
                loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()

            # Update the progress bar description
            progress_bar.set_postfix(loss=running_loss/len(val_loader), accuracy=total_correct/total*100)

    val_loss = running_loss / len(val_loader)
    val_accuracy = total_correct / total * 100
    return val_loss, val_accuracy

def test_model(model, test_loader, device):
    model.eval()
    total_correct = 0
    total = 0
    all_preds = []
    all_labels = []

    # Create a progress bar
    progress_bar = tqdm(test_loader, desc="Testing", leave=False)
    
    with torch.no_grad():
        for data, labels in progress_bar:
            data, labels = data.to(device), labels.to(device)
            with autocast():
                outputs, _, _ = model(data)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())

            # Update the progress bar description
            progress_bar.set_postfix(accuracy=total_correct/total*100)

    test_accuracy = total_correct / total * 100
    return test_accuracy, all_preds, all_labels

def main(num_points, num_classes, batch_size, epochs, dataset, dir_path, download):
    data_dir = f'{dir_path}/data'
    run_name = f"pointnet_{dataset}_{num_points}_{batch_size}_{epochs}"

    os.makedirs(f"{dir_path}", exist_ok=True)
    os.makedirs(f"{dir_path}/{run_name}", exist_ok=True)
    os.makedirs(f"{dir_path}/{run_name}/results", exist_ok=True)

    # Create dataset instances for train, validation, and test
    if dataset == "modelnet10":
        train_dataset = ModelNet10Dataset(root_dir=data_dir,download=download, num_points=num_points, split='train', split_ratio=0.8)
        val_dataset = ModelNet10Dataset(root_dir=data_dir, download=download,num_points=num_points, split='validate', split_ratio=0.8)
        test_dataset = ModelNet10Dataset(root_dir=data_dir, download=download,num_points=num_points, split='test', split_ratio=0.8)
    elif dataset == "modelnet40":
        train_dataset = ModelNet40Dataset(root_dir=data_dir,download=download, num_points=num_points, split='train', split_ratio=0.8)
        val_dataset = ModelNet40Dataset(root_dir=data_dir,download=download, num_points=num_points, split='validate', split_ratio=0.8)
        test_dataset = ModelNet40Dataset(root_dir=data_dir, download=download,num_points=num_points, split='test', split_ratio=0.8)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Create model, loss, and optimizer instances
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    best_model = None
    best_acc = -math.inf

    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, device, optimizer, criterion,scaler)
        val_loss, val_accuracy = validate(model, val_loader, device, criterion)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_model = model.state_dict()

        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}%")
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%")
    
    save_plot_loss_acc(train_accuracies, val_accuracies, run_name=run_name, dir_path=dir_path)

    test_accuracy, all_preds, all_labels = test_model(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy}%")

    save_confusion_matrix(all_labels, all_preds, categories=train_dataset.categories, run_name=run_name, dir_path=dir_path)

    torch.save(best_model, f"{dir_path}/{run_name}/pointnet_model.pth")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train PointNet on ModelNet dataset.')
    parser.add_argument('--num_points', type=int, default=2048, help='number of points in each sample')
    parser.add_argument('--num_classes', type=int, default=40, help='number of output classes')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--dataset', type=str, default='modelnet10', help='dataset to use (modelnet10 or modelnet40)')
    parser.add_argument('--dir_path', type=str, default='/content/drive/MyDrive/pointnet_torch', help='directory path for saving models and results')
    parser.add_argument('--download', type=int, help='download the dataset')

    args = parser.parse_args()
    main(args.num_points, args.num_classes, args.batch_size, args.epochs, args.dataset, args.dir_path, args.download==1)
