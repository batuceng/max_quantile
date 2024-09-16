from scripts.train import train
from scripts.utils import prepare_training

if __name__ == "__main__":
    config = prepare_training()  # This handles both config loading and argument parsing
    print(config)
    train(config)
