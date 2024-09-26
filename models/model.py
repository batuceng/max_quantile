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
        if len(x.shape) == 3:
            x = x.squeeze(0)
            
        return x
    
    # Delete the prototypes in the given indices
    # indices : (0,3,5)
    # new_indices = 1=: 0 , 2=: 1 , 4=: 2
    @torch.no_grad()
    def remove_proto(self, indices):
        mask = torch.full((len(self.head.weight),), True, dtype=bool)
        mask[indices] = False
        new_weights = self.head.weight.data[mask]
        new_biases = self.head.bias.data[mask]

        self.proto_count = new_biases.shape[0]
        new_head = nn.Linear(self.hidden_size, self.proto_count)
        new_head.weight.data = new_weights
        new_head.bias.data = new_biases
        self.head = new_head
        
    # Repeat the prototypes in the given indices and concat to the end
    @torch.no_grad()
    def add_proto(self, indices):
        mask = torch.full((len(self.head.weight),), False, dtype=bool)
        mask[indices] = True
        new_weights = torch.vstack((self.head.weight.data, self.head.weight.data[mask]))
        new_biases = torch.cat((self.head.bias.data, self.head.bias.data[mask]))
        
        self.proto_count = new_biases.shape[0]
        new_head = nn.Linear(self.hidden_size, self.proto_count)
        new_head.weight.data = new_weights
        new_head.bias.data = new_biases
        self.head = new_head