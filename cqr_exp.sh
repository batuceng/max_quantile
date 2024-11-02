#!/bin/bash

data_path="/home/halil/max_quantile/data/raw"

# datasets=("bike_sharing" "concrete" "energy_efficiency" "MEPS_19" "MEPS_20" "MEPS_21" "unconditional_1d" "white_wine")
datasets=("Unconditional_2d_data_outlier_100" "Unconditional_2d_data_outlier_1000")


mkdir ./cqr_results
for dataset in "${datasets[@]}"; do
    echo "Running dataset: $dataset" > "$output_file"  # Overwrite existing file
    for index in {0..9}; do
        current_data_path="$data_path/$dataset/seed_$index/all_data.npy"
        echo "Processing seed $index"  >> "$output_file"
        python3 cqr_main.py --alpha 0.1 --dataset_path  "$current_data_path"  >> "./cqr_results/${dataset}_cqr_0.9.txt" 2>&1
        python3 cqr_main.py --alpha 0.5 --dataset_path  "$current_data_path"  >> "./cqr_results/${dataset}_cqr_0.5.txt" 2>&1
        python3 cqr_main.py --alpha 0.9 --dataset_path  "$current_data_path"  >> "./cqr_results/${dataset}_cqr_0.1.txt" 2>&1
    done
done

