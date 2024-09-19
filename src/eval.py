import torch
# from models.model import RegressionModel
from data.dataset import CustomDataset
from torch.utils.data import DataLoader
import numpy as np
from src.utils import load_config
import torch.nn.functional as F

def eval_model(config, model, quantizer, folder, alpha=0.9,):
    # Load calibration dataset and data loader
    calib_dataset = CustomDataset(config['dataset_path'], mode='cal')
    bs = config['train']['batch_size'] if config['train']['batch_size'] != -1 else len(calib_dataset)
    calib_data_loader = DataLoader(calib_dataset, batch_size=bs, shuffle=False)

    # Load test dataset and data loader
    test_dataset = CustomDataset(config['dataset_path'], mode='test')
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
        test_inputs_list = []
        for x_batch, y_batch in test_data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            test_logits = model(x_batch)  # Model outputs log densities
            test_logits_list.append(test_logits)
            test_labels_list.append(y_batch)
            test_inputs_list.append(x_batch)
        test_logits = torch.cat(test_logits_list, dim=0)  # Shape: [n_test, num_prototypes]
        test_labels = torch.cat(test_labels_list, dim=0)   # Shape: [n_test, label_dim]
        test_inputs = torch.cat(test_inputs_list, dim=0)   # Shape: [n_test, input_dim]

    # Quantize the test labels using the quantizer
    test_mindist, test_proto_indices = quantizer.quantize(test_labels)  # Indices of nearest prototypes

    # Adjust test logits with Voronoi region areas to get adjusted test log probabilities
    adjusted_test_logits = test_logits + torch.log(region_areas.unsqueeze(0))  # Shape: [n_test, num_prototypes]
    adjusted_test_probs = F.softmax(adjusted_test_logits, dim=1)  # Convert to probabilities

    # Sort test logits (log densities) and get sorted indices
    test_logits_sorted, sorted_indices = torch.sort(test_logits, dim=1, descending=True)  # [n_test, num_prototypes]

    # Get region areas sorted accordingly
    region_areas_sorted = region_areas[sorted_indices]  # [n_test, num_prototypes]

    # Adjust the sorted log densities with log of region areas to get adjusted log probabilities
    adjusted_log_probs_sorted = F.softmax(test_logits_sorted + torch.log(region_areas_sorted), dim=1)  # [n_test, num_prototypes]

    # Create a mask where adjusted probabilities >= quantile threshold
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

    # Check the input_dim and output_dim
    input_dim = test_inputs.shape[1]
    output_dim = test_labels.shape[1]

    # Check the input_dim and output_dim
    input_dim = test_inputs.shape[1]
    output_dim = test_labels.shape[1]

    if output_dim == 1:
        save_path = folder + f"/marginal_distribution_{alpha}.png"
        visualize_1d(test_labels, adjusted_test_probs, quantizer, mask, sorted_indices, correct_predictions, coverage, quantile_threshold,save_path)
    elif output_dim == 2:
        save_path = folder + f"/marginal_distribution_{alpha}.png"
        visualize_2d(test_labels, adjusted_test_probs, quantizer, mask, sorted_indices, correct_predictions, coverage, quantile_threshold,save_path)

    return coverage, pinaw_score

def visualize_1d(test_labels, adjusted_test_probs, quantizer, mask, sorted_indices, correct_predictions, coverage, quantile_threshold,save_path):
    """
    Visualize the marginal distribution of p(y) for 1D outputs.
    """
    import matplotlib.pyplot as plt

    # Get prototypes and their corresponding probabilities
    protos = quantizer.protos.detach().cpu().numpy().flatten()  # Shape: [num_prototypes]
    adjusted_probs = adjusted_test_probs.detach().cpu().numpy()  # Shape: [n_test, num_prototypes]
    test_labels_np = test_labels.detach().cpu().numpy().flatten()  # Shape: [n_test]

    # Aggregate probabilities over all test samples
    mean_probs = np.mean(adjusted_probs, axis=0)  # Shape: [num_prototypes]

    # Get the coverage indicator
    covered = correct_predictions.cpu().numpy().astype(bool)

    # Plot the marginal distribution
    plt.figure(figsize=(10, 6))
    plt.bar(protos, mean_probs, width=0.05, alpha=0.7, label='Mean Predicted Probability')

    # Plot the prediction intervals
    included_protos = protos[np.any(mask.cpu().numpy(), axis=0)]
    plt.axvspan(included_protos.min(), included_protos.max(), color='lightgreen', alpha=0.3, label='Prediction Interval')

    # Plot the test labels
    plt.scatter(test_labels_np[covered], np.zeros_like(test_labels_np[covered]), c='blue', marker='x', label='Covered Test Labels')
    plt.scatter(test_labels_np[~covered], np.zeros_like(test_labels_np[~covered]), c='red', marker='x', label='Uncovered Test Labels')

    plt.title(f'Marginal Distribution of p(y) - 1D Output\nCoverage: {coverage:.2f}%')
    plt.xlabel('y')
    plt.ylabel('Probability')
    plt.legend()
    
    plt.savefig(save_path)
    print(f"Saved the plot at {save_path}")

def visualize_2d(test_labels, adjusted_test_probs, quantizer, mask, sorted_indices, correct_predictions, coverage, quantile_threshold,save_path):
    """
    Visualize the marginal distributions of p(y) for 2D outputs.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Get prototypes and their corresponding probabilities
    protos = quantizer.protos.detach().cpu().numpy()  # Shape: [num_prototypes, 2]
    adjusted_probs = adjusted_test_probs.detach().cpu().numpy()  # Shape: [n_test, num_prototypes]
    test_labels_np = test_labels.detach().cpu().numpy()  # Shape: [n_test, 2]

    # Aggregate probabilities over all test samples
    mean_probs = np.mean(adjusted_probs, axis=0)  # Shape: [num_prototypes]

    # Get the coverage indicator
    covered = correct_predictions.cpu().numpy().astype(bool)

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the marginal probability distribution as points sized by mean_probs
    sc = ax.scatter(protos[:, 0], protos[:, 1], mean_probs * 100, c=mean_probs, cmap='viridis', s=mean_probs * 1000, alpha=0.7, label='Mean Predicted Probability')

    # Plot the test labels
    ax.scatter(test_labels_np[covered, 0], test_labels_np[covered, 1], np.zeros_like(test_labels_np[covered, 0]), c='blue', marker='x', label='Covered Test Labels')
    ax.scatter(test_labels_np[~covered, 0], test_labels_np[~covered, 1], np.zeros_like(test_labels_np[~covered, 0]), c='red', marker='x', label='Uncovered Test Labels')

    # Add color bar
    fig.colorbar(sc, ax=ax, label='Mean Probability')

    ax.set_title(f'Marginal Distribution of p(y) - 2D Output\nCoverage: {coverage:.2f}%')
    ax.set_xlabel('y1')
    ax.set_ylabel('y2')
    ax.set_zlabel('Probability (%)')
    ax.legend()
    plt.savefig(save_path)




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
