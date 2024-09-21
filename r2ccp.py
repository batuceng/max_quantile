# Import the model
from R2CCP.main import R2CCP
from data.dataset import CustomDataset
import numpy as np
import os

dataset = "Concrete_Compressive_Strength"
train_dataset = CustomDataset(f"raw/{dataset}/all_data.npy",mode="train")
X_train, y_train = train_dataset.data_x, train_dataset.data_y
scaler_x, scaler_y = train_dataset.scaler_x, train_dataset.scaler_y
cal_dataset = CustomDataset(f"raw/{dataset}/all_data.npy",mode="cal")
X_cal, y_cal = cal_dataset.data_x, cal_dataset.data_y
# Instiantiate the model
fname = "logs/R2CCP/model_save_destination.pth"
if os.path.isfile(fname):
    os.remove(fname)
model = R2CCP({'model_path':'logs/R2CCP/model_save_destination.pth', 'max_epochs':5,
               'optimizer': 'adamw', 'lr':1e-4, 'weight_decay':1e-4,
               'ffn_num_layers':4, 'ffn_hidden_dim':100,
               'loss_weight':1., 'entropy_weight':0.2})
# // model_path is where to save the trained model output (required parameter)

# Fit against the data
model.fit(X_train, y_train, X_cal, y_cal, scaler_x, scaler_y)


test_dataset = CustomDataset(f"raw/{dataset}/all_data.npy",mode="test")
X_test, Y_test = test_dataset.data_x, test_dataset.data_y
# Analyze the results
intervals = model.get_intervals(X_test)
coverage, length = model.get_coverage_length(X_test, Y_test)
print(f"R2CCP Original Code results - dataset:{dataset}")
print(f"1-alpha:{0.90:.2f}, Coverage: {np.mean(coverage)}, Length: {np.mean(length)}")

# If you don't have labels, you can just use get_length
length = model.get_length(X_test)

# Get model predictions
predictions = model.predict(X_test)

# You can also change the desired coverage level
model.set_coverage_level(.5)
coverage, length = model.get_coverage_length(X_test, Y_test)
print(f"1-alpha:{0.50:.2f}, Coverage: {np.mean(coverage)}, Length: {np.mean(length)}")

# You can also change the desired coverage level
model.set_coverage_level(.1)
coverage, length = model.get_coverage_length(X_test, Y_test)
print(f"1-alpha:{0.10:.2f}, Coverage: {np.mean(coverage)}, Length: {np.mean(length)}")
