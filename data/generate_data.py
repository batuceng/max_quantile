#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import os
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Generate data for the experiments')
    parser.add_argument('--dataset', type=str, default='Unconditional_2d_data', help='Name of the dataset to generate')
    parser.add_argument('--seed', type=int, default=0, help='Seed for generating the data')
    return parser.parse_args()



#%%

# Centers of the Gaussians
def unconditional_2d_data_generator(seed):
  
  # create always the same data
  np.random.seed(42)
  
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

  savedir = f'./raw/Unconditional_2d_data/seed_{seed}'
  os.makedirs(savedir, exist_ok=True) 
  np.save(os.path.join(savedir, 'pdf.npy'), np.stack([X,Y,Z]))

  # split the data
  
  train_x, test_x, train_y, test_y = train_test_split(np.zeros((len(data_samples),1)), data_samples, test_size=0.2, random_state=seed)
  test_x, cal_x, test_y, cal_y = train_test_split(test_x, test_y, test_size=0.5, random_state=seed)

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
  
  
def prepare_any_dataset(dataset_id, name,seed): 
    import numpy as np
    import pandas as pd
    import os
    import joblib
    

    dataset = fetch_ucirepo(id=dataset_id) 
    X = dataset.data.features
    y = dataset.data.targets

    # Split the data
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=seed)
    test_x, cal_x, test_y, cal_y = train_test_split(test_x, test_y, test_size=0.5, random_state=seed)

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
    
    
    savedir = f'./raw/{name}/seed_{seed}'
    os.makedirs(savedir, exist_ok=True)
    
    with open(f'./raw/{name}/metadata.txt', 'w') as f:
        f.write(f'Train Data Shape: {train_x.shape}, {train_y.shape}\n')
        f.write(f'Test Data Shape: {test_x.shape}, {test_y.shape}\n')
        f.write(f'Calibration Data Shape: {cal_x.shape}, {cal_y.shape}\n')
    
    
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
  

def unconditional_1d_data_generator(seed):
  np.random.seed(42)
  # Generate samples from N(1, 3)
  data1 = np.random.normal(loc=0.75, scale=0.05, size=11000)
  # Generate samples from N(4, 1)
  data2 = np.random.normal(loc=0, scale=0.002, size=1)
  # Generate samples from N(7, 2)
  data3 = np.random.normal(loc=-0.75, scale=0.05, size=9000)
  # Combine the data
  Y = np.concatenate([data1, data2, data3])
  X = np.zeros((len(Y), 1))
  
  # Split the data
  train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=seed)
  test_x, cal_x, test_y, cal_y = train_test_split(test_x, test_y, test_size=0.5, random_state=seed)
  
  # Normalize the data
  scaler_x = StandardScaler()
  scaler_y = StandardScaler()
  train_x = scaler_x.fit_transform(train_x)
  train_y = scaler_y.fit_transform(train_y.reshape(-1, 1))
  test_x = scaler_x.transform(test_x)
  test_y = scaler_y.transform(test_y.reshape(-1, 1))
  cal_x = scaler_x.transform(cal_x)
  cal_y = scaler_y.transform(cal_y.reshape(-1, 1))
  
  # Save the data
  savedir = f'./raw/Unconditional_1d_data/seed_{seed}'
  os.makedirs(savedir, exist_ok=True)
  # np.save(os.path.join(savedir, 'data.npy'), np.stack([X,Y]))
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





def prepare_meps_data(seed):

  ##############################################################################
  # MEPS 19
  ##############################################################################
  # Load the processed meps_19_reg.csv, extract features X and response y
  df = pd.read_csv('meps_19_reg.csv')
  column_names = df.columns
  response_name = "UTILIZATION_reg"
  column_names = column_names[column_names!=response_name]
  column_names = column_names[column_names!="Unnamed: 0"]

  col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT15F', 'REGION=1',
            'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
            'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
            'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
            'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
            'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
            'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
            'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
            'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
            'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
            'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
            'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
            'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
            'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
            'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
            'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
            'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
            'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
            'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
            'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
            'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
            'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
            'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
            'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
            'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
            'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
            'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']

  y = df[response_name].values
  X = df[col_names].values

  # Split the data
  train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=seed)
  test_x, cal_x, test_y, cal_y = train_test_split(test_x, test_y, test_size=0.5, random_state=seed)
  
  # Normalize the data
  scaler_x = StandardScaler()
  scaler_y = StandardScaler()
  train_x = scaler_x.fit_transform(train_x)
  train_y = scaler_y.fit_transform(train_y.reshape(-1, 1))
  test_x = scaler_x.transform(test_x)
  test_y = scaler_y.transform(test_y.reshape(-1, 1))
  cal_x = scaler_x.transform(cal_x)
  cal_y = scaler_y.transform(cal_y.reshape(-1, 1))
  
  # Save the data
  savedir = f'./raw/MEPS_19/seed_{seed}'
  os.makedirs(savedir, exist_ok=True)
  # np.save(os.path.join(savedir, 'data.npy'), np.stack([X,Y]))
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
  
  # Save the metadata of the dataset
  with open(f'./raw/MEPS_19/metadata.txt', 'w') as f:
    f.write(f'Train Data Shape: {train_x.shape}, {train_y.shape}\n')
    f.write(f'Test Data Shape: {test_x.shape}, {test_y.shape}\n')
    f.write(f'Calibration Data Shape: {cal_x.shape}, {cal_y.shape}\n')
    
    
  ##############################################################################
  # MEPS 20
  ##############################################################################

  # Load the processed meps_20_reg.csv, extract features X and response y
  df = pd.read_csv('meps_20_reg.csv')
  column_names = df.columns
  response_name = "UTILIZATION_reg"
  column_names = column_names[column_names!=response_name]
  column_names = column_names[column_names!="Unnamed: 0"]

  col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT15F', 'REGION=1',
            'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
            'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
            'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
            'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
            'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
            'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
            'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
            'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
            'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
            'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
            'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
            'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
            'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
            'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
            'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
            'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
            'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
            'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
            'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
            'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
            'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
            'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
            'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
            'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
            'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
            'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']


  y = df[response_name].values
  X = df[col_names].values

  # Split the data
  train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=seed)
  test_x, cal_x, test_y, cal_y = train_test_split(test_x, test_y, test_size=0.5, random_state=seed)
  
  # Normalize the data
  scaler_x = StandardScaler()
  scaler_y = StandardScaler()
  train_x = scaler_x.fit_transform(train_x)
  train_y = scaler_y.fit_transform(train_y.reshape(-1, 1))
  test_x = scaler_x.transform(test_x)
  test_y = scaler_y.transform(test_y.reshape(-1, 1))
  cal_x = scaler_x.transform(cal_x)
  cal_y = scaler_y.transform(cal_y.reshape(-1, 1))
  
  # Save the data
  savedir = f'./raw/MEPS_20/seed_{seed}'
  os.makedirs(savedir, exist_ok=True)
  # np.save(os.path.join(savedir, 'data.npy'), np.stack([X,Y]))
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
  # Save the metadata of the dataset
  with open(f'./raw/MEPS_20/metadata.txt', 'w') as f:
    f.write(f'Train Data Shape: {train_x.shape}, {train_y.shape}\n')
    f.write(f'Test Data Shape: {test_x.shape}, {test_y.shape}\n')
    f.write(f'Calibration Data Shape: {cal_x.shape}, {cal_y.shape}\n')
  
    
  ##############################################################################
  # MEPS 21
  ##############################################################################

  # Load the processed meps_21_reg.csv, extract features X and response y
  df = pd.read_csv('meps_21_reg.csv')
  column_names = df.columns
  response_name = "UTILIZATION_reg"
  column_names = column_names[column_names!=response_name]
  column_names = column_names[column_names!="Unnamed: 0"]

  col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT16F', 'REGION=1',
            'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
            'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
            'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
            'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
            'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
            'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
            'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
            'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
            'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
            'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
            'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
            'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
            'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
            'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
            'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
            'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
            'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
            'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
            'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
            'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
            'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
            'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
            'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
            'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
            'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
            'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']


  y = df[response_name].values
  X = df[col_names].values

    # Split the data
    # Split the data
  
  
  train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=seed)
  test_x, cal_x, test_y, cal_y = train_test_split(test_x, test_y, test_size=0.5, random_state=seed)

  # Normalize the data
  scaler_x = StandardScaler()
  scaler_y = StandardScaler()
  train_x = scaler_x.fit_transform(train_x)
  train_y = scaler_y.fit_transform(train_y.reshape(-1, 1))
  test_x = scaler_x.transform(test_x)
  test_y = scaler_y.transform(test_y.reshape(-1, 1))
  cal_x = scaler_x.transform(cal_x)
  cal_y = scaler_y.transform(cal_y.reshape(-1, 1))

  # Save the data
  savedir = f'./raw/MEPS_21/seed_{seed}'
  os.makedirs(savedir, exist_ok=True)
  # np.save(os.path.join(savedir, 'data.npy'), np.stack([X,Y]))
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

  # Save the metadata of the dataset
  with open(f'./raw/MEPS_21/metadata.txt', 'w') as f:
    f.write(f'Train Data Shape: {train_x.shape}, {train_y.shape}\n')
    f.write(f'Test Data Shape: {test_x.shape}, {test_y.shape}\n')
    f.write(f'Calibration Data Shape: {cal_x.shape}, {cal_y.shape}\n')
      

  
# 275 id check

# %%
if __name__ == '__main__':
  
  # args = parse_args()

  for seed in range(10):
    # unconditional_2d_data_generator(seed)
    # print("Saved 2d Unconditional Data!")
    # prepare_any_dataset(165,'Concrete_Compressive_Strength',seed)
    # print("Saved 1d Concrete_Compressive_Strength Data!")
    # prepare_any_dataset(174,'Parkinsons',seed)
    # print("Saved 1d Parkinsons Data!")
    # prepare_any_dataset(186,'White_Wine',seed)
    # print("Saved 1d White_Wine Data!")
    # prepare_any_dataset(242,'Energy_Efficiency',seed)
    # print("Saved 2d Energy_Efficiency Data!")
    # prepare_any_dataset(89,'Solar_Flare',seed)
    # print("Saved 2d Solar_Flare Data!")
    # unconditional_1d_data_generator(seed)
    # print("Saved 1d Unconditional Data!")
    prepare_meps_data(seed)
    print("Saved MEPS Data!")

# %%
