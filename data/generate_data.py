#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import os
from ucimlrepo import fetch_ucirepo 

np.random.seed(0)


#%%
# Centers of the Gaussians
def unconditional_2d_data_generator():
  means = [(0, 0), 
          (3, 3), 
          (-3, -4)]  
  # Covariance matrices
  covariances = [ [[1, 0], 
                  [0, 1]], 
                [[0.5, 0.2], 
                  [0.2, 0.5]], 
                [[0.7, -0.2], 
                  [-0.2, 0.5]] ] 
  # Number of data samples
  sizes = [10000,
          10000,
          10000]

  # Grid space (-7,7)x(-7,7)
  x = np.linspace(-7, 7, 200)
  y = np.linspace(-7, 7, 200)
  X, Y = np.meshgrid(x, y)
  pos = np.dstack((X, Y))

  # Create the combined Gaussian distribution
  Z = np.zeros(X.shape)
  data_samples = []
  for mean, cov, size in zip(means, covariances, sizes):
      # Generate data samples from the Gaussian
      rv = np.random.multivariate_normal(mean, cov, size=size)
      data_samples.append(rv)
      
      # Add the Gaussian PDF contribution to the grid
      rv_pdf = multivariate_normal(mean, cov).pdf(pos)
      # Z += np.exp(-0.5 * (np.sum(np.dot(pos - mean, np.linalg.inv(cov)) * (pos - mean), axis=2))) * np.linalg.det(cov)**(-1/2) * size
      Z += rv_pdf * size  # Weight by the number of samples for each distribution

  data_samples = np.vstack(data_samples)

  # Normalize the combined distribution
  Z /= np.sum(Z)

  savedir = './raw/Unconditional_2d_data'
  os.makedirs(savedir, exist_ok=True) 
  np.save(os.path.join(savedir, 'pdf.npy'), np.stack([X,Y,Z]))

  # split the data
  from sklearn.model_selection import train_test_split
  train_x, test_x, train_y, test_y = train_test_split(np.zeros((len(data_samples),1)), data_samples, test_size=0.2, random_state=1)
  test_x, cal_x, test_y, cal_y = train_test_split(test_x, test_y, test_size=0.5, random_state=1)

  all_data = {
    'train_x': train_x,
    'train_y': train_y,
    'test_x': test_x,
    'test_y': test_y,
    'cal_x': cal_x,
    'cal_y': cal_y,
  }

  np.save(os.path.join(savedir, 'all_data.npy'), all_data)

  # Plot the 2D PDF
  plt.figure(figsize=(8, 8))
  plt.contourf(X, Y, Z, levels=20, cmap='viridis')
  plt.colorbar()
  plt.title('2D PDF of Multiple Gaussians')
  plt.xlabel('X-axis')
  plt.ylabel('Y-axis')
  plt.savefig(os.path.join(savedir, 'pdf.pdf'))
  plt.show()

  # Plot the 2D PDF with data samples
  plt.figure(figsize=(8, 8))
  plt.contourf(X, Y, Z, levels=20, cmap='viridis')
  plt.colorbar()
  plt.scatter(data_samples[:,0], data_samples[:,1], c='r', s=1, marker='x', alpha=0.5)
  plt.title('2D PDF of Multiple Gaussians with Data Samples')
  plt.xlabel('X-axis')
  plt.ylabel('Y-axis')
  plt.savefig(os.path.join(savedir, 'pdf_sampled.pdf'))
  plt.show()

def prepare_concrete_dataset():
  concrete_compressive_strength = fetch_ucirepo(id=165) 
  # data (as pandas dataframes) 
  X = concrete_compressive_strength.data.features 
  y = concrete_compressive_strength.data.targets 
  
  # split the data
  from sklearn.model_selection import train_test_split
  train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)
  test_x, cal_x, test_y, cal_y = train_test_split(test_x, test_y, test_size=0.5, random_state=1)
    
  all_data = {
    'train_x': train_x,
    'train_y': train_y,
    'test_x': test_x,
    'test_y': test_y,
    'cal_x': cal_x,
    'cal_y': cal_y,
  }
  savedir = './raw/Concrete_Compressive_Strength'
  os.makedirs(savedir, exist_ok=True)
  np.save(os.path.join(savedir, 'all_data.npy'), all_data)
  



# %%
