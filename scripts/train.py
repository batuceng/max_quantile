import numpy as np
import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.model import ProtoClassifier
from data.dataset import CustomDataset
from scripts.utils import prepare_training
from quantizers.quantizer import VoronoiQuantizer

def train(config):
    # Load dataset
    # dataset = CustomDataset()
    dataset = CustomDataset(config['dataset_path'])
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    device = torch.device('cpu')

    # Initialize model, loss function, and optimizer
    model = ProtoClassifier(config['input_dim'], config['output_dim'], config['proto_count_per_dim']).to(device)
    criterion = getattr(nn, config['loss_fn'])()
    optimizer = getattr(optim, config['optimizer'])(model.parameters(), lr=config['learning_rate'])
    quantizer = VoronoiQuantizer(dataset.data_y, config['proto_count_per_dim']).to(device)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(config['log_dir'], 'run_{}'.format(config['run_id'])))

    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        protos = quantizer.get_protos().to(device)
        for i, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            quantized_targets = quantizer.quantize(targets).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, quantized_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        writer.add_scalar('Loss/train', running_loss / len(data_loader), epoch)

    writer.close()

