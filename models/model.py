import torch
import torch.nn as nn
import numpy as np

# class RegressionModel(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(RegressionModel, self).__init__()
#         self.fc = nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         return self.fc(x)
    
class ProtoClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, proto_count_per_dim):
        super(ProtoClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proto_count = proto_count_per_dim**output_dim
        self.hidden_size = 256
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        self.head = nn.Linear(self.hidden_size, self.proto_count)
        
    def forward(self, x):
        x = self.net(x)
        x = self.head(x)
        return x
    
    # Delete the prototypes in the given indices
    def remove_proto(self, indices):
        mask = np.full(len(self.head.weight),True,dtype=bool)
        mask[indices] = False
        self.head.weight = self.head.weight[mask]
        self.head.bias = self.head.bias[mask]

    # Repeat the prototypes in the given indices and concat to the end
    def add_proto(self, indices):
        mask = np.full(len(self.head.weight),False,dtype=bool)
        mask[indices] = True
        self.head.weight = torch.vstack(self.head.weight, self.head.weight[mask])
        self.head.bias = torch.vstack(self.head.bias, self.head.bias[mask])
        