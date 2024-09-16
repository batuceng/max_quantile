import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.model import ProtoClassifier
from data.dataset import CustomDataset
from scripts.utils import prepare_training
import os

def train(config):
    # Load dataset
    # dataset = CustomDataset()
    dataset = CustomDataset(config['dataset_path'])
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # Initialize model, loss function, and optimizer
    model = ProtoClassifier(config['input_dim'], config['output_dim'], config['proto_count_per_dim'])
    criterion = getattr(nn, config['loss_fn'])()
    optimizer = getattr(optim, config['optimizer'])(model.parameters(), lr=config['learning_rate'])

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(config['log_dir'], 'run_{}'.format(config['run_id'])))

    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(data_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        writer.add_scalar('Loss/train', running_loss / len(data_loader), epoch)

    writer.close()

