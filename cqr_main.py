

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
# from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.model import ProtoClassifier
from data.dataset import CustomDataset,CustomDataloader
from src.utils import prepare_training, remove_param_from_optimizer
from quantizers.quantizer import VoronoiQuantizer,GridQuantizer
from src.eval import eval_model, get_prototype_usage_density
from src.losses import mindist_loss, repulsion_loss, softmin_grads, distance_based_ce, entropy_loss
import time 
import yaml

def argparse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('dataset_path', type=str, help='Path to dataset')
    parser.add_argument('device', type=str, help='Device to use')   

    return parser.parse_args()


def train():
    # Load dataset
    
    args = argparse()
    
    
    train_dataset = CustomDataset(args.dataset_path,mode='train',device = config['device'])
    bs = -1
    train_data_loader = CustomDataloader(train_dataset, batch_size=bs, shuffle=False)
    
    device = args.device
    model = QRegressor().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for step in range(3000):
        #yq1, yq2, yq3 = model(x)
        #loss = criterion(input=(yq1, yq2, yq3), target=y)
        for i, (train_x_tr, train_y_tr) in enumerate(train_data_loader):
            train_x_tr = train_x_tr.to(device)
            train_y_tr = train_y_tr.to(device)
            yqs = model(train_x_tr)

            loss = PinballLoss(prediction=yqs, target =train_y_scalar,q_low=0.05,q_high=0.95)
            #print(yqs.shape, train_y_tr.shape)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step+1) % 1000 == 0:
                """
                Show your intermediate results
                """
                print('epoch {} loss={:.4}'.format(step+1, loss.data.item()))
                pass
    
    
        
    calib_dataset = CustomDataset(args.dataset_path,mode='calib',device = config['device'])
    






def calculate_calibration_metrics(calib_dataset,alpha,model,device):
    alpha = 0.1
    # Get the prediction
    y_pred = model(calib_dataset.data_x.to(device))

    res_lows = y_pred[:,0:1,0] - calib_dataset.data_y
    res_high = calib_dataset.data_y - y_pred[:,1:2,0] 

    residuals = torch.max(torch.stack([res_lows, res_high],dim=1).squeeze(),dim=1)[0].detach().cpu().numpy()


    adapted_alpha = (1-alpha) * (1 + 1/len(calib_dataset.data_x))
    # # Calculate the quantile residuals
    quantile_residuals = np.quantile(residuals, adapted_alpha, axis=0)
    
    return quantile_residuals


    
    





# 3-headed output
# modification to MyModel
#    return output1, output2, output3 -> return self.net(x)  # torch.nn.Linear(30,3)
class QRegressor(torch.nn.Module):
    def __init__(self, in_size=1, out_size=1):
        super(QRegressor,self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        hidden_layer = 64

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.in_size, hidden_layer),
            torch.nn.LeakyReLU(),

            torch.nn.Linear(hidden_layer, hidden_layer),
            torch.nn.LeakyReLU(),

            torch.nn.Linear(hidden_layer, 2*self.out_size)
            )

    def forward(self,x):
        x = self.net(x)
        b_size = x.shape[0]
        return x.reshape(b_size,2,self.out_size)


def PinballLoss(prediction, target,q_low,q_high):
    '''
    For each sample, the loss is:
    (q_low)*(y-f(x)) if y < f(x)
    (q_high)*(y-f(x)) if y >= f(x)
    where q_low and q_high are quantiles of interest
    '''    
    
    dim_num = prediction.shape[2]
    
    losses = []
    for i in range(dim_num):
        pred = prediction[:,:,i]
        t = target[:,i:i+1]
        e_low =  t - pred[:,0:1]  # !!! if input[:,0]  -> shape = (1000,)
        e_high = t - pred[:,1:2] 
        eq_low = torch.max(q_low*e_low, (q_low-1)*e_low)
        eq_high = torch.max(q_high*e_high, (q_high-1)*e_high)
        loss = (eq_low + eq_high).mean()
        losses.append(loss)

    return torch.stack(losses).mean()

