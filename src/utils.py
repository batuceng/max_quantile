import numpy as np
import random
import os

import yaml
import argparse

import torch

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load the configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_arguments():
    """Parse command-line arguments for overriding config values."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/default_config.yaml', help="Path to the config file.")
    parser.add_argument('--quantizer_type', type=str, help="Override the learning rate in the config.")
    parser.add_argument('--learning_rate', type=float, help="Override the learning rate in the config.")
    parser.add_argument('--quant_learning_rate', type=float, help="Override the quant learning rate in the config.")
    parser.add_argument('--epochs', type=int, help="Override the number of epochs in the config.")
    parser.add_argument('--batch_size', type=int, help="Override the batch size in the config.")
    parser.add_argument('--run_id', type=int, default=1, help="Unique identifier for the training run.")
    
    parser.add_argument('--proto_split_density_threshold', type=float, help="Override the split density threshold in the config.")
    parser.add_argument('--proto_remove_density_threshold', type=float, help="Override the remove density threshold in the config.")
    parser.add_argument('--add_remove_usage_mode', type=str, help="Override the usage mode in the config.")
    parser.add_argument('--repulsion_loss_margin', type=float, help="Override the repulsion loss margin in the config.")
    parser.add_argument('--add_remove_every_n_epoch', type=int, help="Override the add/remove every n epoch in the config.")
    
    parser.add_argument('--device', type=str, default='cuda', help="Device to use for training.")
    parser.add_argument('--dataset_path', type=str, default = None, help="Path to the dataset.")
    
    # quantizer : proto_split_density_threshold, proto_remove_density_threshold, ,usage_mode
    # losses: repulsion_loss_margin
    args = parser.parse_args()

    return args

def update_config_with_args(config, args):
    """Update the config with the values provided by command-line arguments."""
    if args.learning_rate is not None:
        config['train']['learning_rate'] = args.learning_rate
    if args.quantizer_type is not None:
        config['quantizer']['quantizer_type'] = args.quantizer_type
    if args.epochs is not None:
        config['train']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['train']['batch_size'] = args.batch_size
    if args.proto_split_density_threshold is not None:
        config['quantizer']['proto_split_density_threshold'] = args.proto_split_density_threshold
    if args.proto_remove_density_threshold is not None:
        config['quantizer']['proto_remove_density_threshold'] = args.proto_remove_density_threshold
    if args.add_remove_usage_mode is not None:
        config['quantizer']['add_remove_usage_mode'] = args.add_remove_usage_mode
    if args.repulsion_loss_margin is not None:
        config['losses']['repulsion_loss_margin'] = args.repulsion_loss_margin        
    if args.add_remove_every_n_epoch is not None:
        config['quantizer']['add_remove_every_n_epoch'] = args.add_remove_every_n_epoch
    if args.device is not None:
        config['device'] = args.device
    if args.dataset_path is not None:
        config['dataset_path'] = args.dataset_path
    if args.quant_learning_rate is not None:
        config['train']['quant_learning_rate'] = args.quant_learning_rate
    
    config['run_id'] = args.run_id
    
    return config

def prepare_training():
    """Prepare the training configuration by loading the config file and parsing CLI arguments."""
    args = parse_arguments()  # Step 1: Parse command-line arguments
    #print(os.getcwd())
    config = load_config(args.config)  # Step 2: Load the YAML configuration file
    config = update_config_with_args(config, args)  # Step 3: Override config with CLI arguments
    return config


def remove_param_from_optimizer(optim, pg_index):
    # Remove corresponding state
    for param in optim.param_groups[pg_index]['params']:
        if param in optim.state:
            del optim.state[param]
    del optim.param_groups[pg_index]