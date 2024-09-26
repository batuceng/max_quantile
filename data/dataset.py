import torch
from torch.utils.data import Dataset
import numpy as np
import torch
import joblib
import os

class CustomDataset(Dataset):
    def __init__(self, data_path,device, mode='train'):
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
            
        self.data_x = torch.tensor(self.data_x, dtype=torch.float32).to(device)
        self.data_y = torch.tensor(self.data_y, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.data_x)

    def get_real_values(self):
        '''
        Denormalize the data and return the real values.
        '''
        return self.scaler_x.inverse_transform(self.data_x), self.scaler_y.inverse_transform(self.data_y)
    def __getitem__(self, idx):
        x, y = self.data_x[idx], self.data_y[idx]
        return x,y
    


class CustomBlockDataset(Dataset):
    def __init__(self, data_path,device, mode='train'):
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
            
        self.data_x = torch.tensor(self.data_x, dtype=torch.float32).unsqueeze(0)
        self.data_y = torch.tensor(self.data_y, dtype=torch.float32).unsqueeze(0)

    def __len__(self):
        return len(self.data_x)

    def get_real_values(self):
        '''
        Denormalize the data and return the real values.
        '''
        return self.scaler_x.inverse_transform(self.data_x), self.scaler_y.inverse_transform(self.data_y)
    def __getitem__(self, idx):
        x, y = self.data_x[idx], self.data_y[idx]
        return x,y



class CustomDataloader:
    def __init__(self, dataset, batch_size=32,shuffle=True):
        self.dataset = dataset
        if shuffle:
            indexes = torch.randperm(len(self.dataset))
        else:
            indexes = torch.arange(len(self.dataset))
            
        batch_size = len(self.dataset) if batch_size == -1 else batch_size
        current_index = 0
        self.batches = []
        while current_index < len(self.dataset):
            if current_index + batch_size > len(self.dataset):
                self.batches.append(indexes[current_index:])
            else:
                self.batches.append(indexes[current_index:current_index+batch_size])
            current_index += batch_size

    def __len__(self):
        return len(self.batches)
    
    def __iter__(self):
        for batch in self.batches:
            yield self.dataset.data_x[batch], self.dataset.data_y[batch]
