#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import os
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler
import joblib


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


  scaler_x = StandardScaler()
  scaler_y = StandardScaler()
  train_x = scaler_x.fit_transform(train_x)
  train_y = scaler_y.fit_transform(train_y)
  test_x = scaler_x.transform(test_x)
  test_y = scaler_y.transform(test_y)
  cal_x = scaler_x.transform(cal_x)
  cal_y = scaler_y.transform(cal_y)
  
  joblib.dump(scaler_x, os.path.join(savedir, 'scaler_x.pkl'))
  joblib.dump(scaler_y, os.path.join(savedir, 'scaler_y.pkl'))

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
def prepare_any_dataset(dataset_id, name): 
    import numpy as np
    import pandas as pd
    import os
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer

    dataset = fetch_ucirepo(id=dataset_id) 
    X = dataset.data.features
    y = dataset.data.targets

    # Split the data
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)
    test_x, cal_x, test_y, cal_y = train_test_split(test_x, test_y, test_size=0.5, random_state=1)

    # Check if train_x has any string values
    if np.any([isinstance(x, str) for x in train_x.values.flatten()]):
        # Convert string columns to categorical and fit transformer on train_x
        train_x, ct = convert_string_columns_to_categorical(train_x)
        
        # Apply the transformer to test_x and cal_x
        test_x = ct.transform(test_x)
        cal_x = ct.transform(cal_x)
        
        # Get feature names
        feature_names = ct.get_feature_names_out()
        
        # Convert to DataFrame
        train_x = pd.DataFrame(train_x, columns=feature_names)
        test_x = pd.DataFrame(test_x, columns=feature_names)
        cal_x = pd.DataFrame(cal_x, columns=feature_names)
    else:
        ct = None

    # Initialize scalers
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    # Scale the data
    train_x = scaler_x.fit_transform(train_x)
    train_y = scaler_y.fit_transform(train_y.values)
    test_x = scaler_x.transform(test_x)
    test_y = scaler_y.transform(test_y.values)
    cal_x = scaler_x.transform(cal_x)
    cal_y = scaler_y.transform(cal_y.values)

    # Prepare data dictionary
    
    all_data = {
        'train_x': train_x,
        'train_y': train_y,
        'test_x': test_x,
        'test_y': test_y,
        'cal_x': cal_x,
        'cal_y': cal_y,
    }
    # Save the metadata of the dataset
    
    
    savedir = f'./raw/{name}'
    os.makedirs(savedir, exist_ok=True)
    
    with open(f'./raw/{name}/metadata.txt', 'w') as f:
        f.write(f'Train Data Shape: {train_x.shape}, {train_y.shape}\n')
        f.write(f'Test Data Shape: {test_x.shape}, {test_y.shape}\n')
        f.write(f'Calibration Data Shape: {cal_x.shape}, {cal_y.shape}\n')
        
    
    
    # Save directory
    
    
    # Save scalers and transformer
    joblib.dump(scaler_x, os.path.join(savedir, 'scaler_x.pkl'))
    joblib.dump(scaler_y, os.path.join(savedir, 'scaler_y.pkl'))
    if ct:
        joblib.dump(ct, os.path.join(savedir, 'column_transformer.pkl'))

    # Save the data
    np.save(os.path.join(savedir, 'all_data.npy'), all_data)
    

def convert_string_columns_to_categorical(df):
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn import __version__ as sklearn_version
    from packaging import version
    import pandas as pd

    # Get the columns with string values
    string_columns = df.select_dtypes(include=['object']).columns

    # Convert the string columns to categorical
    df[string_columns] = df[string_columns].astype('category')

    # Get the categorical columns
    categorical_columns = df.select_dtypes(include=['category']).columns

    # Version check for OneHotEncoder parameter
    if version.parse(sklearn_version) >= version.parse("1.0"):
        onehotencoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    else:
        onehotencoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    # Apply OneHotEncoder to the categorical columns
    ct = ColumnTransformer(
        transformers=[
            ('onehot', onehotencoder, categorical_columns)
        ],
        remainder='passthrough'
    )

    df_transformed = ct.fit_transform(df)

    # Get the new feature names after one-hot encoding
    feature_names = ct.get_feature_names_out()

    # Return the transformed dataframe and the transformer
    df_transformed = pd.DataFrame(df_transformed, columns=feature_names)

    return df_transformed, ct
  




# %%
if __name__ == '__main__':
  
  if not os.path.exists('./raw/Unconditional_2d_data'):
    unconditional_2d_data_generator()
    print("Saved 2d Unconditional Data!")
  if not os.path.exists('./raw/Concrete_Compressive_Strength'):
    prepare_any_dataset(165,'Concrete_Compressive_Strength')
    print("Saved 1d Concrete_Compressive_Strength Data!")
  if not os.path.exists('./raw/Parkinsons'):
    prepare_any_dataset(174,'Parkinsons')
    print("Saved 1d Parkinsons Data!")
  if not os.path.exists('./raw/White_Wine'):
    prepare_any_dataset(186,'White_Wine')
    print("Saved 1d White_Wine Data!")
  if not os.path.exists('./raw/Energy_Efficiency'):
    prepare_any_dataset(242,'Energy_Efficiency')
    print("Saved 2d Energy_Efficiency Data!")
  if not os.path.exists('./raw/Solar_Flare'):
    prepare_any_dataset(89,'Solar_Flare')
    print("Saved 2d Solar_Flare Data!")
  



# %%
