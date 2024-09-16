import torch
import torch.nn as nn

# class RegressionModel(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(RegressionModel, self).__init__()
#         self.fc = nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         return self.fc(x)
    
class ProtoClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, proto_count):
        super(ProtoClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proto_count = proto_count
        self.hidden_size = 256
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.proto_count),
        )
        self.protos = nn.Parameter(torch.random.randn((self.proto_count, self.output_dim)), requires_grad=False)
        
    def forward(self, x):
        x = self.net(x)
        return x
    