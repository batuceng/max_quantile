import itertools
import subprocess
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed,ProcessPoolExecutor
from tqdm import tqdm  # Import tqdm for the progress bar
import argparse

# Define the hyperparameters
add_remove_every_n_epoch_list = [50, 100]
proto_split_density_threshold_list = [0.001, 0.002, 0.01]
proto_remove_density_threshold_list = [0.0001, 0.0002, 0.001]
repulsion_loss_margin_list = [1e-2, 2e-3, 1e-3, 1e-4]
batch_size_list = [-1, 256]
epoch_list = [150, 300, 500]
seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


# add_remove_every_n_epoch_list = [50]
# proto_split_density_threshold_list = [0.001]
# proto_remove_density_threshold_list = [0.0001]
# repulsion_loss_margin_list = [1e-2]
# batch_size_list = [-1]
# epoch_list = [150]
# seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def arg_parse():
    parser = argparse.ArgumentParser(description='MaxQuantile Grid Search')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the config file')
    parser.add_argument('--results_path', type=str, required=True, help='Path to save the results CSV')

    args = parser.parse_args()
    return args

args = arg_parse()
dataset_path = args.dataset_path
config_path = args.config_path
results_path = args.results_path

# Generate all combinations of hyperparameters
hyperparameter_combinations = list(itertools.product(
    add_remove_every_n_epoch_list,
    proto_split_density_threshold_list,
    proto_remove_density_threshold_list,
    repulsion_loss_margin_list,
    batch_size_list,
    epoch_list,
    seed_list
))

# Number of devices (assuming 4 GPUs)
num_devices = 4

# Max number of parallel processes
max_workers = 16  # Adjust based on your system capabilities
# Ensure the results directory exists
results_dir = os.path.dirname(results_path)
if results_dir != '' and not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Prepare the list of all experiments to run
all_experiments = []
for index, hyperparams in enumerate(hyperparameter_combinations):
    device = f'cuda:{index % num_devices}'
    all_experiments.append(hyperparams + (device,))

def run_experiment(params):
    (
        add_remove_every_n_epoch,
        proto_split_density_threshold,
        proto_remove_density_threshold,
        repulsion_loss_margin,
        batch_size,
        epochs,
        seed,
        device
    ) = params
    dataset_seed_path = f'{dataset_path}/seed_{seed}/all_data.npy'

    command = [
        'python', '/home/halil/max_quantile/main.py',
        '--add_remove_every_n_epoch', str(add_remove_every_n_epoch),
        '--proto_split_density_threshold', str(proto_split_density_threshold),
        '--proto_remove_density_threshold', str(proto_remove_density_threshold),
        '--repulsion_loss_margin', str(repulsion_loss_margin),
        '--device', str(device),
        '--dataset_path', dataset_seed_path,
        '--config', config_path,
        '--batch_size', str(batch_size),
        '--epochs', str(epochs),
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy()
        )
        output = result.stdout.strip()
        error_output = result.stderr.strip()

        # Check for errors
        if result.returncode != 0:
            print(f'Error in subprocess (seed {seed}, device {device}):\n{error_output}')
            # Return an empty list to indicate failure
            return []

        # Print output for debugging (optional)
        # print(f'Output:\n{output}')

        # Parse multiple lines of output
        lines = output.strip().split('\n')
        results = []

        # first line is the experiment_path 
        experiment_path = lines.pop(0).strip()
        
        for idx, line in enumerate(lines):
            if line.strip() == '':
                continue  # Skip empty lines
            try:
                cov_value, PINAW_value = line.strip().split(',')
                cov_value = cov_value.strip()
                PINAW_value = PINAW_value.strip()
                result_tuple = (
                    add_remove_every_n_epoch,
                    proto_split_density_threshold,
                    proto_remove_density_threshold,
                    repulsion_loss_margin,
                    batch_size,
                    epochs,
                    seed,
                    experiment_path,
                    cov_value,
                    PINAW_value
                )
                results.append(result_tuple)
            except ValueError:
                print(f'Unexpected line format from main.py (seed {seed}, device {device}): {line}')
                # You can choose to skip this line or handle it as needed
                continue

        return results  # Return the list of result tuples

    except Exception as e:
        print(f'Exception running experiment with params {params}: {e}')
        return []  # Return an empty list to indicate failure

# Prepare the CSV file
csv_file = open(results_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
# Write the header
csv_writer.writerow([
    'add_remove_every_n_epoch',
    'proto_split_density_threshold',
    'proto_remove_density_threshold',
    'repulsion_loss_margin',
    'batch_size',
    'epochs',
    'seed',
    'experiment_path',
    'cov_value',
    'PINAW_value'
])

# Run experiments in parallel with tqdm progress bar
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(run_experiment, params) for params in all_experiments]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Experiments", unit="exp"):
        results = future.result()
        for result in results:
            csv_writer.writerow(result)