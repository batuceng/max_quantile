import torch
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_path, mode='train', transform_x=None,transform_y=None):
        assert mode in ['train', 'test', 'cal']
        all_data = np.load(data_path, allow_pickle=True).item()
        self.mode = mode
        
        if transform_x is None and mode == 'train':
            self.mean = np.mean(all_data['train_x'], axis=0)
            self.std = np.std(all_data['train_x'], axis=0)
            self.std[self.std == 0] = 1
            transform_x = lambda x: (x - self.mean) / self.std
        self.transform_x = transform_x
        
        if transform_y is None and mode == 'train':
            self.mean_y = np.mean(all_data['train_y'], axis=0)
            self.std_y = np.std(all_data['train_y'], axis=0)
            self.std_y[self.std_y == 0] = 1
            transform_y = lambda x: (x - self.mean_y) / self.std_y
        self.transform_y = transform_y
        
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
        
        if self.transform_x:
            x = self.transform_x(x)
        if self.transform_y:
            y = self.transform_y(y)
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
  