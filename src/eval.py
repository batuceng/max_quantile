import torch
# from models.model import RegressionModel
from data.dataset import CustomDataset
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from matplotlib.collections import PolyCollection

@torch.no_grad()
def inference(dataloader, model, device):
    # Process test data
    log_density_preds_list = []
    targets_list = []
    inputs_list = []
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        log_density_preds = model(x_batch)  # Model outputs log densities
        log_density_preds_list.append(log_density_preds)
        targets_list.append(y_batch)
        inputs_list.append(x_batch)
    log_density_preds = torch.cat(log_density_preds_list, dim=0)  # Shape: [n_test, num_prototypes]
    targets = torch.cat(targets_list, dim=0)   # Shape: [n_test, label_dim]
    inputs = torch.cat(inputs_list, dim=0)   # Shape: [n_test, input_dim]
    return log_density_preds, targets, inputs

def eval_model(config, model, quantizer, folder, alpha=0.9,mode = "prob_th"):
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
    region_areas = quantizer.get_areas_pyvoro()  # Should return areas corresponding to prototypes
    region_areas = torch.tensor(region_areas, dtype=torch.float32).to(device)

    # Log densities, real y values, and conditions for calibration
    cal_log_density_preds, cal_targets, cal_inputs = inference(calib_data_loader, model, device)
    # Quantize the calibration labels using the quantizer
    cal_mindist, cal_proto_indices = quantizer.quantize(cal_targets)  # Indices of nearest prototypes

    # Adjust calibration logits with Voronoi region areas to get log probabilities
    cal_log_prob_preds = cal_log_density_preds + torch.log(region_areas.unsqueeze(0))  # Shape: [n_cal, num_prototypes]
    cal_prob_preds = torch.softmax(cal_log_prob_preds, dim=1)  # Convert to log probabilities
    
    # 1. Calculate qhat_prob by prob, expand regions until qhat_prob reached sorted by probs (Calib: %90, %85 -> %85 - Max Prob First, threshould by prob)
    # 1. Calculate qhat_prob by prob, expand regions until qhat_prob reached sorted by densities (Calib: %90, %85 -> %85 - Max Dens First, threshould by prob)
    # 1. Calculate qhat_dens by prob, expand regions until qhat_prob reached sorted by densities (Calib: %90, 12 -> 12 - Max Dens First, threshould by prob)
    # mode = "prob_th"
    print(f"Conformal mode: {mode}")
    mode in ["prob_th", "dens_th"]
    if mode == "prob_th":
        # cal_pi = torch.argsort(cal_prob_preds, dim=1, descending=True)  # [n_cal, num_prototypes] # sort the probabilities according to the density values
        cal_pi = torch.argsort(cal_log_density_preds, dim=1, descending=True)  # [n_cal, num_prototypes] # sort the probabilities according to the density values
        cal_prob_preds_sorted = torch.gather(cal_prob_preds, 1, cal_pi)  # [n_cal, num_prototypes] # sort the probabilities according to the density values
        cal_prob_cumsum = torch.cumsum(cal_prob_preds_sorted, dim=1)  # [n_cal, num_prototypes] # Cumulative sum of probabilities all samples are 1 at the end
        cal_proto_indices_in_sorted = cal_pi.argsort(dim=1)[torch.arange(len(cal_prob_preds)), cal_proto_indices] # true class index in the sorted array
        cal_scores = cal_prob_cumsum[torch.arange(len(cal_prob_preds)), cal_proto_indices_in_sorted] # values of the true class in the cumulative sum
        n = cal_targets.shape[0] # number of samples
        quantile_threshold = np.quantile(cal_scores.detach().cpu().numpy(), np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher") # quantile threshold
    elif mode == "dens_th":
        # cal_log_density_preds_sorted, cal_log_density_preds_sorted_indices = torch.sort(cal_log_density_preds, dim=1, descending=True)
        # cal_proto_indices_in_sorted = cal_log_density_preds_sorted_indices.argsort(dim=1)[torch.arange(len(cal_prob_preds)), cal_proto_indices] # true class index in the sorted array
        # cal_scores = cal_log_density_preds_sorted[torch.arange(len(cal_log_density_preds)), cal_proto_indices_in_sorted] # values of the true class in the cumulative sum
        cal_scores = cal_log_density_preds[torch.arange(len(cal_log_density_preds)), cal_proto_indices]
        n = cal_targets.shape[0] # number of samples
        # alpha = 1-alpha
        quantile_threshold = np.quantile(cal_scores.detach().cpu().numpy(), np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher") # quantile threshold


    ### TEST ###
    test_log_density_preds, test_targets, test_inputs = inference(test_data_loader, model, device)
    # Quantize the test labels using the quantizer
    test_mindist, test_proto_indices = quantizer.quantize(test_targets)  # Indices of nearest prototypes
    # Adjust test logits with Voronoi region areas to get adjusted test log probabilities
    test_log_prob_preds = test_log_density_preds + torch.log(region_areas.unsqueeze(0))  # Shape: [n_test, num_prototypes]
    test_prob_preds = F.softmax(test_log_prob_preds, dim=1)  # Convert to probabilities
    
    if mode == "prob_th":
        # test_pi = torch.argsort(test_prob_preds, dim=1, descending=True)  # [n_test, num_prototypes] # sort the probabilities according to the density values
        test_pi = torch.argsort(test_log_density_preds, dim=1, descending=True)  # [n_test, num_prototypes] # sort the probabilities according to the density values
        test_prob_preds_sorted = torch.gather(test_prob_preds, 1, test_pi)  # [n_test, num_prototypes] # sort the probabilities according to the density values
        
        test_prob_cumsum = torch.cumsum(test_prob_preds_sorted, dim=1)  # [n_test, num_prototypes] # Cumulative sum of probabilities all samples are 1 at the end
        prediction_set = torch.where(test_prob_cumsum <= quantile_threshold,test_pi,-1)   # [n_test, num_prototypes] # mask for the prediction set
    elif mode == "dens_th":
        prediction_set = torch.where(test_log_density_preds>=quantile_threshold, torch.arange(test_log_density_preds.shape[1], device=device), -1)
    # Calculate the coverage
    coverage = torch.mean((prediction_set == test_proto_indices.unsqueeze(1)).sum(dim=1).float())
    print(f"Coverage of the data by the conformal prediction set: {100*coverage:.2f}%")

    region_set = []
    for i in range(len(prediction_set)):
        unique_set = np.setdiff1d(prediction_set[i].detach().cpu().numpy(),[-1])
        region_area = region_areas[unique_set].sum().detach().cpu().numpy()
        region_set.append(region_area)
    
    pinaw_score = np.mean(region_set)
    print(f"PINAW Score: {pinaw_score:.5f}")
    # Check the input_dim and output_dim
    input_dim = test_inputs.shape[1]
    output_dim = test_targets.shape[1]

    # Check the input_dim and output_dim
    input_dim = test_inputs.shape[1]
    output_dim = test_targets.shape[1]

    if output_dim == 1:
        visualize_protos_1d(quantizer, cal_targets,prediction_set, folder + f"/protos_{alpha}.png")
        # Visualize a few samples
        visualize_test_sample_1d(quantizer, test_log_density_preds, test_targets, test_inputs, prediction_set, idx=5, save_path=folder + f"/sample_logdens_{5}_alpha_{alpha}.png", ytitle="Log Density")
        visualize_test_sample_1d(quantizer, test_log_density_preds, test_targets, test_inputs, prediction_set, idx=15, save_path=folder + f"/sample_logdens_{15}_alpha_{alpha}.png", ytitle="Log Density")
        visualize_test_sample_1d(quantizer, test_log_density_preds, test_targets, test_inputs, prediction_set, idx=25, save_path=folder + f"/sample_logdens_{25}_alpha_{alpha}.png", ytitle="Log Density")
        visualize_test_sample_1d(quantizer, test_prob_preds, test_targets, test_inputs, prediction_set, idx=5, save_path=folder + f"/sample_prob_{5}_alpha_{alpha}.png", ytitle="Probability Dist")
        visualize_test_sample_1d(quantizer, test_prob_preds, test_targets, test_inputs, prediction_set, idx=15, save_path=folder + f"/sample_prob_{15}_alpha_{alpha}.png", ytitle="Probability Dist")
        visualize_test_sample_1d(quantizer, test_prob_preds, test_targets, test_inputs, prediction_set, idx=25, save_path=folder + f"/sample_prob_{25}_alpha_{alpha}.png", ytitle="Probability Dist")
    #     visualize_1d(test_targets, test_prob_preds, quantizer, mask, sorted_indices, correct_predictions, coverage, quantile_threshold,save_path)
    # elif output_dim == 2:
    #     save_path = folder + f"/marginal_distribution_{alpha}.png"
    #     visualize_2d(test_targets, test_prob_preds, quantizer, mask, sorted_indices, correct_predictions, coverage, quantile_threshold,save_path)
    elif output_dim == 2:
        save_path = folder + f"/marginal_distribution_{alpha}.png"
        visualize_y_marginal_with_voronoi(quantizer, test_targets, prediction_set, save_path)

    return coverage, pinaw_score


def visualize_test_sample_1d(quantizer, visual_variable, test_targets, test_inputs, prediction_set, idx, save_path, ytitle):
    """
    Visualize a test sample output with bar plot of predicted log densities,
    scatter plot of prototypes, target values marked with 'x', and hlines for selected prototypes.
    
    Args:
        quantizer: The quantizer object containing prototypes and region areas.
        visual_variable: Variable to visualize, It can be log density, density, probability, area.
        test_targets: The actual target values for the test samples.
        test_inputs: The input data for the test samples (not used here).
        prediction_set: The set of selected prototypes for additional highlighting (optional).
        idx: The index of the sample to visualize.
        save_path: Path where to save the generated plot.
    """
    # Extract prototypes as numpy array
    protos_np = quantizer.get_protos_numpy()  # Assuming this method returns prototypes in a numpy array
    region_areas = quantizer.get_areas_pyvoro()  # Assuming this method returns areas in a numpy array
    decision_boundaries = quantizer.get_proto_decision_boundaries()
    
    sample_visual_variable = visual_variable[idx]  # Log density predictions for the sample
    sample_target = test_targets[idx].item()  # Target values for the sample
    sample_prediction_set = prediction_set[idx]
    
    plt.figure(figsize=(10, 6))
    
    # Plot the predicted log density as bar plots within decision boundaries
    for i, proto in enumerate(protos_np):
        height = sample_visual_variable[i]  # Use the predicted log density as the height of the bar
        left, right = decision_boundaries[i]
        width = right - left
        center = (right + left) / 2
        if i in sample_prediction_set:
            plt.bar(center, height.item(), width=width.item(), align='center', alpha=0.5, label='Log Density' if i == 0 else None, color='red')
        else:
            plt.bar(center, height.item(), width=width.item(), align='center', alpha=0.5, label='Log Density' if i == 0 else None, color='blue')

    # Scatter plot for prototypes
    plt.scatter(protos_np, np.zeros_like(protos_np), label='Prototypes', color='black', marker='D', zorder=5)

    # Plot target values marked with 'x'
    plt.scatter(sample_target, np.zeros_like(sample_target), label='Target Value', color='green', marker='x', zorder=6)

    # Add titles and labels
    plt.title(f"Sample {idx} Visualization")
    plt.xlabel('Prototypes')
    plt.ylabel(f"{ytitle} / Target Value")
    plt.legend()
    
    # Save the figure
    plt.savefig(save_path)
    # Show the figure
    plt.close()
    
def visualize_y_marginal_with_voronoi(quantizer, test_targets, prediction_set, save_path):
    # Plot the probability distribution over the 2D grid using the bin edges
    proto_centers = quantizer.get_protos_numpy()
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)  # nrows, ncols, index

    highlight_color = 'green'  # Color for top_indices regions

    # Create the colormap for other regions
    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(vmin=test_targets.min(), vmax=test_targets.max())
    prediction_set_unique = torch.unique(prediction_set)
    
    polygons = quantizer.get_proto_decision_boundaries()
    colors = []
    for i, polygon in enumerate(polygons):    
        if i in prediction_set_unique:
            color = highlight_color  # Use the highlight color for top regions
        else:
            color = cmap(0)
        
        ax.fill(*zip(*polygon), color = color, alpha=0.5)
    
    
    ax.scatter(proto_centers[:, 0], proto_centers[:, 1], marker='x', c='red', s=1)
    ax.set_xlim(proto_centers[:, 0].min() - 0.1, proto_centers[:, 0].max() + 0.1)
    ax.set_ylim(proto_centers[:, 1].min() - 0.1, proto_centers[:, 1].max() + 0.1)
    ax.set_title('Probability Distribution over the 2D Space with Voronoi Tessellation')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Probability')

    plt.show()
    plt.savefig(save_path)

def visualize_protos_1d(quantizer, cal_labels, prediction_set, save_path):
    protos_np = quantizer.get_protos_numpy()
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(cal_labels.flatten().detach().cpu().numpy(), bins=50)
    plt.scatter(protos_np.flatten(), [0.1]*len(protos_np),c='red', marker='x', s=100)
    plt.savefig(save_path)

def get_prototype_usage_density(train_data_loader, model, quantizer, usage_mode, device):
    log_density_preds, targets, inputs = inference(train_data_loader, model, device)
    adjacencies, proto_areas = quantizer.get_adjancencies_and_volumes()
    qdist, quantized_target_index = quantizer.quantize(targets)
    soft_quantized_target_index = quantizer.soft_quantize(targets)
    if usage_mode == "bincountbased":
        prototype_usage = torch.bincount(quantized_target_index, minlength=quantizer.protos.size(0))
    elif usage_mode == "softlabelbased":
        prototype_usage = soft_quantized_target_index.sum(dim=0)
    else: raise NotImplementedError(f"No mode {usage_mode}")
    prototype_usage_density = prototype_usage / prototype_usage.sum()
    return prototype_usage_density