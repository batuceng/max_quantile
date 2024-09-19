from src.train import train
from src.utils import prepare_training, seed_everything

if __name__ == "__main__":
    config = prepare_training()  # This handles both config loading and argument parsing
    seed_everything(config['seed'])
    train(config)
