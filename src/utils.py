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
    parser.add_argument('--learning_rate', type=float, help="Override the learning rate in the config.")
    parser.add_argument('--epochs', type=int, help="Override the number of epochs in the config.")
    parser.add_argument('--batch_size', type=int, help="Override the batch size in the config.")
    parser.add_argument('--run_id', type=int, default=1, help="Unique identifier for the training run.")
    args = parser.parse_args()

    return args

def update_config_with_args(config, args):
    """Update the config with the values provided by command-line arguments."""
    if args.learning_rate is not None:
        config['train']['learning_rate'] = args.learning_rate
    if args.epochs is not None:
        config['train']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['train']['batch_size'] = args.batch_size
    config['run_id'] = args.run_id
    return config

def prepare_training():
    """Prepare the training configuration by loading the config file and parsing CLI arguments."""
    args = parse_arguments()  # Step 1: Parse command-line arguments
    print(os.getcwd())
    config = load_config(args.config)  # Step 2: Load the YAML configuration file
    config = update_config_with_args(config, args)  # Step 3: Override config with CLI arguments
    return config
