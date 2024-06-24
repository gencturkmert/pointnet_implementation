import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import math

from data.modelnet10 import *
from data.modelnet40 import *
from model.pointnet import *
from utils.plot import *

# Parameters
num_points = 2048  # number of points in each sample
num_classes = 40   # number of output classes
batch_size = 32
epochs = 100
dataset = "modelnet10"

run_name = f"pointnet_{dataset}_{num_points}_{batch_size}_{epochs}"

# Create dataset instances for train, validation, and test
train_dataset = ModelNet10Dataset(root_dir='ModelNet10', num_points=num_points, split='train', split_ratio=0.8)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

val_dataset = ModelNet10Dataset(root_dir='ModelNet10', num_points=num_points, split='validate', split_ratio=0.8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

test_dataset = ModelNet10Dataset(root_dir='ModelNet10', num_points=num_points, split='test', split_ratio=0.8)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Create model, loss, and optimizer instances
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PointNet(num_points=num_points, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()  # appropriate loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_one_epoch(epoch_index, train_loader):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for batch_idx, data in enumerate(train_loader):
        points, labels = data['points'].to(device), data['labels'].to(device)
        optimizer.zero_grad()
        outputs, _, _ = model(points)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % 10 == 9:  # print every 10 mini-batches
            print('[Epoch: %d, Batch: %5d] loss: %.3f' %
                  (epoch_index + 1, batch_idx + 1, running_loss / 10))
            running_loss = 0.0

    print('Epoch %d completed in %.2f seconds.' % (epoch_index + 1, time.time() - start_time))

def train_one_epoch(model, train_loader, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total = 0
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, _, _ = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = total_correct / total * 100
    return train_loss, train_accuracy


def validate(model, val_loader, device, criterion):
    model.eval()
    total_correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            outputs, _, _ = model(data)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()

    val_loss = running_loss / len(val_loader)
    val_accuracy = total_correct / total * 100
    return val_loss, val_accuracy

def test_model(model, test_loader, device):
    model.eval()
    total_correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs, _, _ = model(data)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())

    test_accuracy = total_correct / total * 100
    return test_accuracy, all_preds, all_labels


train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []
best_model = None
best_acc = -math.inf

for epoch in range(epochs):
    train_loss, train_accuracy = train_one_epoch(model, train_loader, device, optimizer, criterion)
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
    

save_plot_loss_acc(train_accuracies, val_accuracies,run_name=run_name)

test_accuracy, all_preds, all_labels = test_model(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy}%")

save_confusion_matrix(all_labels, all_preds, categories=train_dataset.categories,run_name=run_name)

torch.save(best_model, f"../results/{run_name}/pointnet_model.pth")