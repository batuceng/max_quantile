#!/bin/bash

# Define the names of the datasets (without the .yaml extension)
names=(
    # "bike_sharing"
    # "energy_efficiency"
    # "MEPS_20"
    # "parkinsons"
    "Unconditional_2d_data_outlier_100"
    "Unconditional_2d_data_outlier_1000"
    # "concrete"
    # "MEPS_19"
    # "MEPS_21"
    # "uncond_1d"
    # "white_wine"
)

# Number of GPUs available
num_gpus=4

# Initialize an array with GPU device IDs
gpu_devices=(0 1 2 3)

# Array of seeds
seeds=({0..9})

# Loop over each dataset name
for name in "${names[@]}"; do
    echo "Processing dataset: $name"

    # Define paths specific to the current dataset
    log_file="results_grid/$name.log"
    # config_path="/home/halil/max_quantile/configs_grid/$name.yaml"
    config_path="/home/halil/max_quantile/configs_grid/unconditional_2d.yaml"
    data_path="/home/halil/max_quantile/data/raw/$name"

    # Remove existing log file if it exists to start fresh
    if [ -f "$log_file" ]; then
        rm "$log_file"
    fi

    # Function to run for each seed
    run_seed() {
        seed=$1
        device_id=$2
        dataset_path="$data_path/seed_$seed/all_data.npy"
        temp_log="temp_log_${name}_seed_$seed.log"

        # Run the Python script with the current dataset path and GPU device, and write output to temp log
        python3 main.py --dataset_path="$dataset_path" \
        --config="$config_path" \
        --device="cuda:$device_id" \
        > "$temp_log" 2>&1
    }

    # Run seeds in parallel, limiting concurrency to the number of GPUs
    running_jobs=0
    for seed in "${seeds[@]}"; do
        device_index=$((seed % num_gpus))
        device_id=${gpu_devices[$device_index]}

        echo "Starting seed $seed on GPU cuda:$device_id for dataset $name"

        run_seed "$seed" "$device_id" &

        ((running_jobs++))

        # If the number of running jobs reaches the number of GPUs, wait for all to finish
        if [ "$running_jobs" -ge "$num_gpus" ]; then
            wait
            running_jobs=0
        fi
    done

    # Wait for any remaining background processes
    wait

    # After all processes are complete, concatenate temp logs into main log in order
    for seed in "${seeds[@]}"; do
        temp_log="temp_log_${name}_seed_$seed.log"
        if [ -f "$temp_log" ]; then
            # Optional: Add a separator in the log file for clarity
            echo "=================== Dataset: $name | Seed $seed ===================" >> "$log_file"
            cat "$temp_log" >> "$log_file"
            rm "$temp_log"
        else
            echo "Temporary log file for dataset $name, seed $seed not found." >> "$log_file"
        fi
    done

    echo "Finished processing dataset: $name"
done