import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.path = './raw/data1.npy'
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def load_data(self):
        X, Y = np.load(self.path)
        
class UnconditionalSynthData(Dataset):
    def __init__(self, X, Y, transform=None):
        self.path = './raw/data1.npy'
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def load_data(self):
        X, Y = np.load(self.path)
        