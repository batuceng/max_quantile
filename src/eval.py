import torch
from models.model import RegressionModel
from data.dataset import CustomDataset
from torch.utils.data import DataLoader
from src.utils import load_config

def evaluate():
    config = load_config('./configs/default_config.yaml')
    dataset = CustomDataset(config['dataset_path'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    model = RegressionModel(config['input_dim'], config['output_dim'])
    model.load_state_dict(torch.load('./model_checkpoint.pth'))  # Load trained model

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            outputs = model(x)
            # Evaluate performance here
