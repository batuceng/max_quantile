import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.model import ProtoClassifier
from data.dataset import CustomDataset
from src.utils import prepare_training
from quantizers.quantizer import VoronoiQuantizer
from src.eval import eval_model
from src.losses import mindist_loss, repulsion_loss, softmin_grads

def train(config):
    # Load dataset
    train_dataset = CustomDataset(config['dataset_path'],mode='train')
    bs = config['train']['batch_size'] if config['train']['batch_size']!=-1 else train_dataset.__len__()
    train_data_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    
    device = torch.device('cuda')

    # Initialize model, loss function, and optimizer
    model = ProtoClassifier(config['model']['input_dim'], config['model']['output_dim'], config['model']['proto_count_per_dim']).to(device)

    cross_entropy_loss = nn.CrossEntropyLoss()
    
    optimizer = getattr(optim, config['train']['optimizer'])(model.parameters(), lr=config['train']['learning_rate'])
    if config['quantizer']['type'] == 'voronoi':
        quantizer = VoronoiQuantizer(train_dataset.data_y, config['model']['proto_count_per_dim']).to(device)
    elif config['quantizer']['type'] == 'grid':
        pass
    else:
        raise NotImplementedError(f"Quantizer type {config['quantizer']['type']} not implemented.")
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(config['log_dir'], 'run_{}'.format(config['run_id'])))

    # Training loop
    for epoch in range(config['train']['epochs']):
        model.train()
        running_loss = 0.0
        # Wrap train_data_loader with tqdm for batch-level progress
        for i, (inputs, targets) in enumerate(tqdm(train_data_loader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}")):
            inputs, targets = inputs.to(device), targets.to(device)
            proto_areas = torch.tensor(quantizer.get_areas()).to(device)
            qdist, quantized_target_index = quantizer.quantize(targets)
            optimizer.zero_grad()
            log_density_preds = model(inputs)
            log_prob_preds = log_density_preds + torch.log(proto_areas) 
            ce_loss = cross_entropy_loss(log_prob_preds / config['losses']['cross_entropy_temperature'], quantized_target_index)
            mindist = mindist_loss(targets, quantizer.protos)
            repulsion = repulsion_loss(quantizer.protos,margin = config["losses"]["repulsion_loss_margin"])
            loss = ce_loss * config['losses']['cross_entropy_weight'] + mindist * config['losses']['mindist_weight'] + repulsion * config['losses']['repulsion_loss_weight']
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # Logging to TensorBoard
        writer.add_scalar('Loss/train', running_loss / len(train_data_loader), epoch)

    eval_model(config, model, train_dataset.transform, quantizer)
    
    writer.close()
