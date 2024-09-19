import torch
from torch.utils.data import Dataset
import numpy as np

import joblib
import os

class CustomDataset(Dataset):
    def __init__(self, data_path, mode='train'):
        '''
        Data is read from a numpy file. The values are already normalized.
        '''
        assert mode in ['train', 'test', 'cal']
        all_data = np.load(data_path, allow_pickle=True).item()
        self.data_path = data_path
        self.mode = mode        
        self.scaler_x = joblib.load(os.path.join(os.path.dirname(self.data_path), 'scaler_x.pkl'))
        self.scaler_y = joblib.load(os.path.join(os.path.dirname(self.data_path), 'scaler_y.pkl'))
        
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

    def get_real_values(self):
        '''
        Denormalize the data and return the real values.
        '''
        return self.scaler_x.inverse_transform(self.data_x), self.scaler_y.inverse_transform(self.data_y)
    
    def __getitem__(self, idx):
        x, y = self.data_x[idx], self.data_y[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
  