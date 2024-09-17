import numpy as np
import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.model import ProtoClassifier
from data.dataset import CustomDataset
from scripts.utils import prepare_training
from quantizers.quantizer import VoronoiQuantizer

def train(config):
    # Load dataset
    dataset = CustomDataset(config['dataset_path'])
    bs = config['batch_size'] if config['batch_size']!=-1 else dataset.__len__()
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)
    
    device = torch.device('cuda')

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

        # Wrap data_loader with tqdm for batch-level progress
        for i, (inputs, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")):
            inputs, targets = inputs.to(device), targets.to(device)
            protos = quantizer.get_protos()
            proto_areas = torch.tensor(quantizer.get_areas()).to(device)
            
            qdist, quantized_targets = quantizer.quantize(targets)
            
            optimizer.zero_grad()
            log_density_preds = model(inputs)
            log_prob_preds = log_density_preds + proto_areas
            loss = criterion(log_prob_preds, quantized_targets)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Logging to TensorBoard
        writer.add_scalar('Loss/train', running_loss / len(data_loader), epoch)

    writer.close()
