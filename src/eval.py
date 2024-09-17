import torch
# from models.model import RegressionModel
from data.dataset import CustomDataset
from torch.utils.data import DataLoader
import numpy as np
from src.utils import load_config
import torch.nn.functional as F

def eval_model(config, model, train_transform, quantizer, alpha=0.9):
    # Load calibration dataset and data loader
    calib_dataset = CustomDataset(config['dataset_path'], mode='cal', transform=train_transform)
    bs = config['train']['batch_size'] if config['train']['batch_size'] != -1 else len(calib_dataset)
    calib_data_loader = DataLoader(calib_dataset, batch_size=bs, shuffle=False)

    # Load test dataset and data loader
    test_dataset = CustomDataset(config['dataset_path'], mode='test', transform=train_transform)
    bs = config['train']['batch_size'] if config['train']['batch_size'] != -1 else len(test_dataset)
    test_data_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Get region areas
    region_areas = quantizer.get_areas()  # Should return areas corresponding to prototypes
    region_areas = torch.tensor(region_areas, dtype=torch.float32).to(device)

    # Process calibration data
    with torch.no_grad():
        cal_logits_list = []
        cal_labels_list = []
        for x_batch, y_batch in calib_data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            cal_logits = model(x_batch)  # Model outputs log densities
            cal_logits_list.append(cal_logits)
            cal_labels_list.append(y_batch)
        cal_logits = torch.cat(cal_logits_list, dim=0)  # Shape: [n_cal, num_prototypes]
        cal_labels = torch.cat(cal_labels_list, dim=0)   # Shape: [n_cal, label_dim]

    # Quantize the calibration labels using the quantizer
    cal_mindist, cal_proto_indices = quantizer.quantize(cal_labels)  # Indices of nearest prototypes

    # Adjust calibration logits with Voronoi region areas to get log probabilities
    adjusted_cal_logits = cal_logits + torch.log(region_areas.unsqueeze(0))  # Shape: [n_cal, num_prototypes]
    adjusted_cal_logits = torch.softmax(adjusted_cal_logits, dim=1)  # Convert to log probabilities
    # Get the log probabilities assigned to the true prototypes
    true_class_log_probs = adjusted_cal_logits[torch.arange(len(adjusted_cal_logits)), cal_proto_indices]

    # Sort the true class log probabilities
    sorted_true_class_log_probs, _ = torch.sort(true_class_log_probs)

    # Compute the quantile threshold
    quantile_threshold = np.quantile(sorted_true_class_log_probs.cpu().numpy(), 1 - alpha)

    # Process test data
    with torch.no_grad():
        test_logits_list = []
        test_labels_list = []
        for x_batch, y_batch in test_data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            test_logits = model(x_batch)  # Model outputs log densities
            test_logits_list.append(test_logits)
            test_labels_list.append(y_batch)
        test_logits = torch.cat(test_logits_list, dim=0)  # Shape: [n_test, num_prototypes]
        test_labels = torch.cat(test_labels_list, dim=0)   # Shape: [n_test, label_dim]

    # Quantize the test labels using the quantizer
    test_mindist, test_proto_indices = quantizer.quantize(test_labels)  # Indices of nearest prototypes
    test_logits_sorted, sorted_indices = torch.sort(test_logits, dim=1, descending=True)  # [n_test, num_prototypes]
    # Get region areas sorted accordingly
    region_areas_sorted = region_areas[sorted_indices]  # [n_test, num_prototypes]
    # Adjust the sorted log densities with log of region areas to get adjusted log probabilities
    adjusted_log_probs_sorted = torch.softmax(test_logits_sorted + torch.log(region_areas_sorted),dim=1)  # [n_test, num_prototypes]
    # Create a mask where adjusted log probabilities >= quantile threshold
    mask = adjusted_log_probs_sorted >= quantile_threshold  # [n_test, num_prototypes], boolean
    # Number of prototypes included per sample
    num_included = mask.sum(dim=1)  # [n_test], integer
    # Find the position of the true prototype in sorted_indices
    true_proto_in_sorted = (sorted_indices == test_proto_indices.unsqueeze(1))  # [n_test, num_prototypes], boolean
    # Get the position index of the true prototype for each sample
    positions_of_true_proto = torch.argmax(true_proto_in_sorted.int(), dim=1)  # [n_test]
    # Correct predictions: True if position of true prototype is before num_included
    correct_predictions = (positions_of_true_proto < num_included).float()  # [n_test]
    # Calculate coverage
    coverage = correct_predictions.mean().item() * 100
    print(f"Coverage of the data by the conformal prediction set: {coverage:.2f}%")
    # Calculate the area of each prediction set
    prediction_set_areas = (region_areas_sorted * mask.float()).sum(dim=1)  # [n_test]
    # Calculate PINAW
    pinaw_score = prediction_set_areas.mean().item()
    print(f"PINAW Score: {pinaw_score:.5f}")
    # Optionally, return the coverage and PINAW score
    return coverage, pinaw_score

# def evaluate():
#     config = load_config('./configs/default_config.yaml')
#     dataset = CustomDataset(config['dataset_path'])
#     dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

#     model = RegressionModel(config['input_dim'], config['output_dim'])
#     model.load_state_dict(torch.load('./model_checkpoint.pth'))  # Load trained model

#     model.eval()
#     with torch.no_grad():
#         for x, y in dataloader:
#             outputs = model(x)
#             # Evaluate performance here
