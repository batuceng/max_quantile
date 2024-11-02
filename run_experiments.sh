#!/bin/bash

# Configuration variable
config="/home/halil/max_quantile/configs/uncond_1d.yaml"

# Dataset path
dataset_path="/home/halil/max_quantile/data/raw/unconditional_1d"

# Output file to store the results temporarily
output_file="unconditional_1d.txt"
: > $output_file  # Clear file

# Iterate through seeds 0 to 9
for seed in {0..9}
do
    # Run the Python script and capture the output
    python3 main.py --dataset_path="$dataset_path/seed_$seed/all_data.npy" \
                    --config="$config" > temp_output.txt

    # Extract the results (last three lines of the output)
    tail -n 3 temp_output.txt >> $output_file
done

# Python script to process results
python3 <<EOF
import numpy as np

# Read the results from the temporary file
with open('$output_file', 'r') as file:
    results = [line.strip().split(',') for line in file.readlines()]

# Convert to numpy array for easier manipulation
results = np.array(results, dtype=float).reshape(-1, 3, 2)  # 10 seeds, 3 cov/pinaw pairs, 2 values each

# Calculate mean and standard deviation for each (cov, pinaw) pair
means = np.mean(results, axis=0)
stds = np.std(results, axis=0)

# Print results in the desired format
for i, coverage in enumerate([0.9, 0.5, 0.1]):
    print(f"{means[i, 0]:.2f},{stds[i, 0]:.2f}")
    print(f"{means[i, 1]:.2f},{stds[i, 1]:.2f}")
EOF

# Clean up temporary files
rm temp_output.txt $output_file