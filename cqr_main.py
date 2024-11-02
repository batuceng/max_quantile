import argparse
import sys
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Assuming these modules are available in your project structure
from models.model import ProtoClassifier
from data.dataset import CustomDataset,CustomDataloader
from src.utils import prepare_training, remove_param_from_optimizer
from quantizers.quantizer import VoronoiQuantizer, GridQuantizer
from src.eval import eval_model, get_prototype_usage_density
from src.losses import mindist_loss, repulsion_loss, softmin_grads, distance_based_ce, entropy_loss
import time
import yaml

import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train QRegressor with Pinball Loss and Conformal Prediction')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cpu or cuda)')
    parser.add_argument('--batch_size', type=int, default=-1, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha value for conformal prediction')
    args = parser.parse_args()
    return args

# Define the QRegressor model
class QRegressor(torch.nn.Module):
    def __init__(self, in_size=1, out_size=1):
        super(QRegressor, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        hidden_layer = 64

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.in_size, hidden_layer),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_layer, hidden_layer),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_layer, 2 * self.out_size)
        )

    def forward(self, x):
        x = self.net(x)
        b_size = x.shape[0]
        return x.view(b_size, 2, self.out_size)

# Define the Pinball Loss function
def PinballLoss(prediction, target, q_low=0.25, q_high=0.75):
    '''
    For each sample, the loss is:
    (q_low)*(y - f(x)) if y < f(x)
    (q_high)*(y - f(x)) if y >= f(x)
    where q_low and q_high are quantiles of interest
    '''
    dim_num = prediction.shape[2]  # Number of output dimensions
    losses = []

    for i in range(dim_num):
        pred = prediction[:, :, i]
        t = target[:, i:i+1]
        e_low = t - pred[:, 0:1]
        e_high = t - pred[:, 1:2]
        eq_low = torch.max(q_low * e_low, (q_low - 1) * e_low)
        eq_high = torch.max(q_high * e_high, (q_high - 1) * e_high)
        loss = (eq_low + eq_high).mean()
        losses.append(loss)

    return torch.stack(losses).mean()

def calculate_calibration_metrics(calib_dataset, alpha, model, device):
    # Move data to device
    x_calib = calib_dataset.data_x.to(device)
    y_calib = calib_dataset.data_y.to(device)

    # Get the predictions
    y_pred = model(x_calib)  # Shape: (N, 2, out_size)

    # Correct residual calculation
    res_low = y_pred[:, 0, :] - y_calib  # Lower bound residuals
    res_high = y_calib - y_pred[:, 1, :]  # Upper bound residuals

    # Compute the nonconformity scores
    residuals = torch.max(res_low, res_high)
    residuals = torch.clamp(residuals, min=0).detach().cpu().numpy()

    # Adjust alpha if necessary
    adjusted_alpha = 1 - alpha  # Use the original alpha or adjust if needed

    # Calculate the quantile of residuals
    quantile_residuals = np.quantile(residuals, adjusted_alpha, axis=0)

    return quantile_residuals  # Shape: (out_size,)

# Function to compute coverage and PINAW on test data
def evaluate_model(test_dataset, model, quantile_residuals, device,dataset_name,desired_covarage):
    # Move data to device
    x_test = test_dataset.data_x.to(device)
    y_test = test_dataset.data_y.to(device)

    # Get the predictions
    y_pred = model(x_test)  # Shape: (N, 2, out_size)

    # Apply conformal adjustment
    y_pred_lower = y_pred[:, 0, :] - torch.tensor(quantile_residuals, device=device)
    y_pred_upper = y_pred[:, 1, :] + torch.tensor(quantile_residuals, device=device)

    # Check for band crossing and correct
    # band_crossing_mask = y_pred_lower > y_pred_upper
    # if band_crossing_mask.any():
    #     print(f"Warning: Band crossings detected and corrected in test predictions.")
    #     y_pred_lower_corrected = torch.where(band_crossing_mask, y_pred_upper, y_pred_lower)
    #     y_pred_upper_corrected = torch.where(band_crossing_mask, y_pred_lower, y_pred_upper)
    #     y_pred_lower = y_pred_lower_corrected
    #     y_pred_upper = y_pred_upper_corrected

    # Calculate coverage
    covered = ((y_pred_lower <= y_test) & (y_test <= y_pred_upper)).all(dim=1)
    coverage = covered.float().mean().item()
    
    # Calculate PINAW
    interval_width = y_pred_upper - y_pred_lower  # Shape: (N, out_size)
    interval_areas = interval_width.prod(dim=1)  # Product over dimensions, shape: (N,)
    
    PINAW = interval_areas.mean(dim=0)  # Mean over samples

    print(f"Test Coverage: {coverage * 100:.2f}%")
    print(f"PINAW: {PINAW:.4f}")
    save_path = "cqr_results/" + dataset_name + "_" +desired_covarage + ".png"
    
    if y_test.shape[1] == 1:
        visualize_1d_model(y_pred_lower.detach().cpu().numpy(),y_pred_upper.detach().cpu().numpy(), y_test.detach().cpu().numpy(),save_path)
    else:
        visualize_2d_model(y_pred_lower.detach().cpu().numpy(),y_pred_upper.detach().cpu().numpy(), y_test.detach().cpu().numpy(),  save_path)
    return coverage, PINAW


def visualize_1d_model(y_pred_lower, y_pred_upper, y_test, save_path="results.png", ytitle="Density"):
    """
    Visualize true vs predicted distribution using histogram and vertical lines for predicted bounds.
    
    Args:
        y_pred_lower: Lower bound of the predicted interval for the test samples.
        y_pred_upper: Upper bound of the predicted interval for the test samples.
        y_test: The actual target values for the test samples.
        save_path: Path where to save the generated plot.
        ytitle: The y-axis title for the histogram plot.
    """
    
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of true values
    # normalize the histogram to get a probability density

    hist, bins = np.histogram(y_test, bins=100)
    hist_normalized = hist / np.sum(hist)

    # Plotting the manually normalized histogram
    plt.bar(bins[:-1], hist_normalized, width=(bins[1] - bins[0]), alpha=0.5, color='blue', label='True')

    # plt.hist(y_test, bins=100, alpha=0.5, color='blue', label='True')

    # Plot vertical lines for the lower and upper bounds
    lower_bound = y_pred_lower[0]
    upper_bound = y_pred_upper[0]
    
    plt.axvline(lower_bound, color='red', linestyle='dashed', linewidth=2, label='Lower Bound')
    plt.axvline(upper_bound, color='red', linestyle='dashed', linewidth=2, label='Upper Bound')
    
    # color the region between the bounds
    plt.fill_betweenx([0, 0.09], lower_bound, upper_bound, color='red', alpha=0.2, label='Prediction Interval')

    # Adding titles and labels
    plt.title('True vs Predicted Distribution')
    plt.xlabel('Values')
    plt.ylabel(ytitle)
    
    # Adding grid and legend
    # plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="upper left")
    
    # Save and show the plot
    plt.savefig(save_path)
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def visualize_2d_model(y_pred_lower, y_pred_upper, y_test,y_train,y_cal, save_path="results_2d.png", ytitle="Probability"):
    """
    Visualize true vs predicted distribution in 2D using scatter plot and shaded region for predicted bounds.
    
    Args:
        y_pred_lower: Lower bound of the predicted interval for the test samples (2D, same length as y_test).
        y_pred_upper: Upper bound of the predicted interval for the test samples (2D, same length as y_test).
        y_test: The actual target values for the test samples (N x 2 array).
        save_path: Path where to save the generated plot.
        ytitle: The y-axis title for the scatter plot.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)  # nrows, ncols, index

    # Plot scatter of true values
    ax.scatter(y_test[:, 0], y_test[:, 1], s=15 , color='blue', alpha=0.9, label='True Values', edgecolor='none')

    # Plot predicted lower and upper bounds as shaded regions
    ax.fill_betweenx([y_pred_lower[0, 1], y_pred_upper[0, 1]], 
                     y_pred_lower[0, 0], y_pred_upper[0, 0], 
                     color='green', alpha=0.5, label='Prediction Interval')

    # Plot the actual lower and upper bounds as dashed lines for X and Y axis limits
    ax.axvline(y_pred_lower[0, 0], color='green', linewidth=2, linestyle='dashed', label='Lower Bound (X)')
    ax.axhline(y_pred_lower[0, 1], color='green', linewidth=2, linestyle='dashed', label='Lower Bound (Y)')
    ax.axvline(y_pred_upper[0, 0], color='green', linewidth=2, linestyle='dashed', label='Upper Bound (X)')
    ax.axhline(y_pred_upper[0, 1], color='green', linewidth=2, linestyle='dashed', label='Upper Bound (Y)')

    # Add titles and labels
    # ax.set_title('Probability Distribution over the 2D Space with Bounds')
    ax.set_xlabel('Y1-axis')
    ax.set_ylabel('Y2-axis')
    
    # Adding grid and legend
    # ax.grid(True, linestyle='--', alpha=0.7)
    # ax.legend()

    plt.show()
    plt.savefig(save_path)
    
    
def main():
    args = parse_arguments()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")

    # Load training dataset
    train_dataset = CustomDataset(args.dataset_path, mode='train', device=device)
    train_loader = CustomDataloader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize the model
    in_size = train_dataset.data_x.shape[1]  # Input feature size
    out_size = train_dataset.data_y.shape[1]  # Output size
    model = QRegressor(in_size=in_size, out_size=out_size).to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    model.train()
    for epoch in range(args.num_epochs):
        epoch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            y_pred = model(batch_x)

            # Compute loss
            loss = PinballLoss(prediction=y_pred, target=batch_y, q_low=args.alpha / 2, q_high= 1 - args.alpha / 2)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        # Print training progress
        # print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {np.mean(epoch_losses):.4f}")

    # Load calibration dataset
    calib_dataset = CustomDataset(args.dataset_path, mode='cal', device=device)

    # Calculate calibration metrics (quantile residuals)
    alpha = args.alpha
    quantile_residuals = calculate_calibration_metrics(calib_dataset, alpha, model, device)

    # Load test dataset
    test_dataset = CustomDataset(args.dataset_path, mode='test', device=device)

    # Evaluate the model on test data
    results = evaluate_model(test_dataset, model, quantile_residuals, device,dataset_name = args.dataset_path.split("/")[-3],desired_covarage = str(1 - alpha))
    

if __name__ == '__main__':
    main()