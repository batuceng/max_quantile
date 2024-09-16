import torch
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_path, mode='train', transform=None):
        assert mode in ['train', 'test', 'cal']
        all_data = np.load(data_path, allow_pickle=True).item()
        self.mode = mode
        self.transform = transform

        if self.mode == 'train':
            self.data_x = all_data['train_x']
            self.data_y = all_data['train_y']
        elif self.mode == 'test':
            self.data_x = all_data['test_x']
            self.data_y = all_data['test_y']
        elif self.mode == 'cal':
            self.data_x = all_data['cal_x']
            self.data_y = all_data['cal_y']

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x, y = self.data_x[idx], self.data_y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
