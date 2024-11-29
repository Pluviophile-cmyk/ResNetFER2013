import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import FER2013
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from Model import BasicBlock, ResNet
from load_data import FER2013Dataset
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

data_directory = 'D://Python//IntroAI//fer2013.csv'
df = pd.read_csv(data_directory)
full_dataset = FER2013Dataset(csv_file=data_directory)

# Calculate lengths for each split (8:1:1 ratio)
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

# ResNet18 and 34 architecture
def ResNet18(num_classes=7):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def ResNet34(num_classes=7):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

model = ResNet34(num_classes=7)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20

train_acc = []
val_acc = []
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    correct_train = 0
    processed_train = 0
    pbar = tqdm(train_loader)

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = criterion(y_pred, target)

        loss.backward()
        optimizer.step()

        pred = y_pred.argmax(dim=1, keepdim=True)
        correct_train += torch.sum(pred.squeeze() == target).item()
        processed_train += len(data)

        pbar.set_description(desc=f'Loss={loss.item():.4f} Accuracy={100 * correct_train / processed_train:.2f}%')

        train_losses.append(loss)

    train_accuracy = 100 * correct_train / processed_train
    train_acc.append(train_accuracy)
    print(f"Epoch {epoch + 1}/{epochs} - Training Accuracy: {train_accuracy:.2f}%")

    model.eval()
    correct_val = 0
    processed_val = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            y_pred = model(data)
            loss = criterion(y_pred, target)

            pred = y_pred.argmax(dim=1, keepdim=True)
            correct_val += torch.sum(pred.squeeze() == target).item()
            processed_val += len(data)

            val_losses.append(loss)
    val_accuracy = 100 * correct_val / processed_val
    val_acc.append(val_accuracy)
    print(f"Epoch {epoch + 1}/{epochs} - Validation Accuracy: {val_accuracy:.2f}%")
