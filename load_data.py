import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class FER2013Dataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)  # Load the CSV file
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-7, 7)),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.491,), (0.247,))
        ])

    def __len__(self):
        return len(self.data_frame)  # Return the total number of samples

    def __getitem__(self, idx):
        # Get the image data and label from the DataFrame
        img_data = self.data_frame.iloc[idx]
        img = np.fromstring(img_data['pixels'], sep=' ').astype(np.uint8).reshape(48, 48)  # Reshape to (48, 48)

        # Convert to PIL Image
        img = Image.fromarray(img)
        label = img_data['emotion']  # Get the label

        if self.transform:
            img = self.transform(img)  # Apply transformations

        return img, label  # Return the image and label

