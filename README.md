# Project Overview

This repository contains the code for training and evaluating models on various datasets using Voronoi quantizers. It supports multiple datasets, custom loss functions, and plots evaluation metrics.

## Folder Structure

- `configs/`: Configuration files for different runs and experiments.
- `data/`: Dataset files and code to load/generate datasets.
- `logs/`: TensorBoard logs for different runs.
- `main.py`: Entry point for running training and evaluation.
- `models/`: Contains model architecture files.
- `quantizers/`: Code for quantizers including Voronoi quantization.
- `src/`: Core code for training, evaluation, utilities, and custom losses.
  - `train.py`: Training logic and pipeline.
  - `eval.py`: Evaluation logic and pipeline.
  - `utils.py`: Utility functions such as data preparation and result logging.
  - `losses.py`: Custom loss functions used in the model.

## To-Do List

### Datasets
- [ ] Add support for different datasets.
  - [ ] Implement dataset loaders for each new dataset in `data/dataset.py`. 
  - [ ] Save any dataset following the same structure of CustomDataset. 
    - It should be saved in `data/raw/{dataset_name}/all_data.npy` as preprocessed. 
    - `all_data.npy` should have a dict of {'train_x','train_y','test_x','test_y','cal_x','cal_y'}
  - [ ] Ensure that datasets are properly integrated into the training pipeline.
  - [ ] Add configuration settings in `default_config.yaml` to select datasets.

### Evaluation
- [ ] Finalize `eval.py` for evaluating the trained models.
  - [ ] Compute Coverage, PINAW...
  - [ ] Generate plots of results for 1d and 2d data.
  - [ ] Save generated `logs` folder of related training loop.
- [ ] Implement evaluation on validation/test datasets during training.

### Loss Functions
- [ ] Add `losses.py` to the `src/` directory.
  - [ ] Implement custom loss functions for the models.
  - [ ] Test new loss functions and integrate them into the training pipeline.

### Refactoring
- [x] Rename `src/` to `src/` for better clarity.
- [x] Move all relevant code under the `src/` directory (training, evaluation, utils, and losses).

### Testing (Optional)
- [ ] Write unit tests for training and evaluation code.
  - [ ] Add tests for `train.py` to verify training pipeline functionality.
  - [ ] Add tests for `eval.py` to check evaluation metrics and correctness.

## Running the Project

To train a model, run the following command:

```bash
python main.py --config configs/default_config.yaml
