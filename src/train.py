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
import time 
import yaml

def train(config):
    # Load dataset
    train_dataset = CustomDataset(config['dataset_path'],mode='train')
    bs = config['train']['batch_size'] if config['train']['batch_size']!=-1 else train_dataset.__len__()
    train_data_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    
    device = torch.device('cuda')

    # Initialize model, loss function, and optimizer
    model = ProtoClassifier(config['model']['input_dim'], config['model']['output_dim'], config['model']['proto_count_per_dim']).to(device)

    cross_entropy_loss = nn.CrossEntropyLoss()
    
    if config['quantizer']['quantizer_type'] == 'voronoi':
        quantizer = VoronoiQuantizer(train_dataset.transform_y(train_dataset.data_y), config['model']['proto_count_per_dim']).to(device)
    elif config['quantizer']['quantizer_type'] == 'grid':
        pass
    else:
        raise NotImplementedError(f"Quantizer type {config['quantizer']['quantizer_type']} not implemented.")
    
    optimizer = getattr(optim, config['train']['optimizer'])(list(model.parameters()) + list(quantizer.parameters()) , lr=config['train']['learning_rate'])
    # it should contain be under the dataset folder and inside the dataset folder it should have the time folder 
    time_str = time.strftime("%Y%m%d-%H%M%S")
    experiement_path = os.path.join(config['log_dir'],config['dataset_path'].split('/')[-2],config['quantizer']['quantizer_type'],time_str)
    os.makedirs(experiement_path,exist_ok=True)
    
    writer = SummaryWriter(log_dir=experiement_path)
    # Save config
    yaml_path = os.path.join(experiement_path,'config.yaml')
    with open(yaml_path, 'w') as file:
        yaml.dump(config, file)

    
    # Training loop
    for epoch in range(config['train']['epochs']):
        model.train()
        running_total_loss = 0.0
        running_CE_loss = 0.0
        running_MinDist_loss = 0.0
        running_Repulsion_loss = 0.0
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
            torch.nn.utils.clip_grad_norm_(quantizer.parameters(),0.2)
            optimizer.step()
            
            running_total_loss += loss.item()
            running_CE_loss += ce_loss.item()
            running_MinDist_loss += mindist.item()
            running_Repulsion_loss += repulsion.item()
            
        # Logging to TensorBoard
        writer.add_scalar('Loss/train/total', running_total_loss / len(train_data_loader), epoch)
        writer.add_scalar('Loss/train/CE', running_CE_loss / len(train_data_loader), epoch)
        writer.add_scalar('Loss/train/mindist', running_MinDist_loss / len(train_data_loader), epoch)
        writer.add_scalar('Loss/train/repulsion', running_Repulsion_loss / len(train_data_loader), epoch)
        writer.add_scalars(
            'Percentage/train', 
            {
            'CE_percent': (running_CE_loss * config['losses']['cross_entropy_weight']) / (running_total_loss)*100,
            'MinDist_percent': (running_MinDist_loss * config['losses']['mindist_weight']) / (running_total_loss)*100,
            'Repulsion_percent': (running_Repulsion_loss * config['losses']['repulsion_loss_weight']) / (running_total_loss)*100,
            },
            epoch)

    
    covarage_01,pinaw_01 = eval_model(config, model, train_dataset.transform_x,train_dataset.transform_y, quantizer,alpha=0.1,folder=experiement_path)
    covarage_05,pinaw_05 = eval_model(config, model, train_dataset.transform_x,train_dataset.transform_y, quantizer,alpha=0.5,folder=experiement_path)
    covarage_09,pinaw_09 = eval_model(config, model, train_dataset.transform_x,train_dataset.transform_y, quantizer,alpha=0.9,folder=experiement_path)
    writer.add_scalar('Coverage/0.1', covarage_01, epoch)
    writer.add_scalar('PINAW/0.1', pinaw_01, epoch)
    writer.add_scalar('Coverage/0.5', covarage_05, epoch)
    writer.add_scalar('PINAW/0.5', pinaw_05, epoch)
    writer.add_scalar('Coverage/0.9', covarage_09, epoch)
    writer.add_scalar('PINAW/0.9', pinaw_09, epoch)
    
    # Save model
    model_path = os.path.join(experiement_path,'model.pth')
    torch.save(model.state_dict(), model_path)
    # save quantizer
    quantizer_path = os.path.join(experiement_path,'quantizer.pth')
    torch.save(quantizer.state_dict(), quantizer_path)
    
    
    writer.close()