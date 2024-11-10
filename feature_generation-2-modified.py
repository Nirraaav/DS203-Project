import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch, MeanShift, SpectralClustering, OPTICS, AffinityPropagation
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import librosa
import matplotlib.pyplot as plt
from librosa.feature.rhythm import tempo as compute_tempo_function
from scipy.stats import skew, kurtosis
from tqdm import tqdm

INDEX = 20

def aggregate_mfcc_selective(mfcc_data):
    # mfcc_selected = mfcc_data[:, :] 
    # select rows 1, 2, and 3 from mfcc_data
    mfcc_selected = mfcc_data[0:3, :]
    
    mfcc_mean = np.mean(mfcc_selected, axis=1)
    mfcc_std = np.std(mfcc_selected, axis=1)
    mfcc_max = np.max(mfcc_selected, axis=1)
    mfcc_min = np.min(mfcc_selected, axis=1)
    
    features = np.concatenate([mfcc_mean, mfcc_std, mfcc_max, mfcc_min])
    return features

def aggregate_mfcc_selective2(mfcc_data):
    mfcc_selected = mfcc_data[0:20, :] 
    
    mfcc_mean = np.mean(mfcc_selected, axis=1)
    mfcc_std = np.std(mfcc_selected, axis=1)
    mfcc_max = np.max(mfcc_selected, axis=1)
    mfcc_min = np.min(mfcc_selected, axis=1)
    
    features = np.concatenate([mfcc_mean, mfcc_std, mfcc_max, mfcc_min])
    return features

# Initialize lists to store features and file names
mfcc_all_songs = []
file_names = []
generated_features = []

# Loop through all CSV files in the data-v2 directory and sort them
data_directory = r'.\Labelled_New_Data2\new-test-data\test-mfcc-v2-copy'
data2_directory = r'.\Labelled_New_Data2\new-test-data\test-mfcc-v2'
# data_directory = 'test-mfcc-copy'
# data2_directory = 'test-mfcc'
files = [file for file in os.listdir(data_directory) if file.endswith(".csv")]
files.sort()  # Sort the files alphabetically

for file_name in tqdm(files):
    file_path = os.path.join(data_directory, file_name)
    mfcc_data = pd.read_csv(file_path, header=None).values
    file2_path = os.path.join(data2_directory, file_name)   
    mfcc_data2 = pd.read_csv(file2_path, header=None).values
    mfcc_data2 = mfcc_data2[1:21,:]

    # print(file_name, os.path.join(data_directory, file_name), os.path.join(data2_directory, file_name))

    # Compute aggregated MFCC features
    aggregated_features = aggregate_mfcc_selective(mfcc_data)
    aggregated_features2 = aggregate_mfcc_selective2(mfcc_data2)
    skewness = skew(mfcc_data, axis=1)
    kurt = kurtosis(mfcc_data, axis=1)
    range_max_min = np.ptp(mfcc_data, axis=1) # peak-to-peak range for each MFCC coefficient from 1 to 20

    delta_mfcc = np.diff(mfcc_data, axis=1)
    delta_delta_mfcc = np.diff(delta_mfcc, axis=1)

    delta_mean = np.mean(delta_mfcc, axis=1)
    delta_std = np.std(delta_mfcc, axis=1)
    delta_max = np.max(delta_mfcc, axis=1)
    delta_min = np.min(delta_mfcc, axis=1)
    delta_skew = skew(delta_mfcc, axis=1)
    delta_kurt = kurtosis(delta_mfcc, axis=1)

    delta_mfcc2 = np.diff(mfcc_data2, axis=1)
    delta_delta_mfcc2 = np.diff(mfcc_data2, axis=1)

    delta_mean2 = np.mean(delta_mfcc2, axis=1)
    delta_std2 = np.std(delta_mfcc2, axis=1)
    delta_max2 = np.max(delta_mfcc2, axis=1)
    delta_min2 = np.min(delta_mfcc2, axis=1)
    delta_skew2 = skew(delta_mfcc2, axis=1)
    delta_kurt2 = kurtosis(delta_mfcc2, axis=1)

    window_size = 10  # Example window size
    mfcc_rolling_mean = pd.DataFrame(mfcc_data).T.rolling(window=window_size).mean().T
    mfcc_rolling_std = pd.DataFrame(mfcc_data).T.rolling(window=window_size).std().T

    mfcc_correlation = np.corrcoef(mfcc_data)

    energy_entropy = -np.sum(mfcc_data ** 2 * np.log(mfcc_data ** 2 + 1e-10), axis=1)
    energy_entropy_delta = -np.sum(delta_mfcc ** 2 * np.log(delta_mfcc ** 2 + 1e-10), axis=1)
    energy_entropy_delta2 = -np.sum(delta_mfcc2 ** 2 * np.log(delta_mfcc2 ** 2 + 1e-10), axis=1)

    # print the length of each feature
    # print(len(aggregated_features), len(aggregated_features2), len(skewness), len(kurt), len(range_max_min), len(delta_mean), len(delta_std), len(delta_max), len(delta_min), len(delta_mean2), len(delta_std2), len(delta_max2), len(delta_min2))
        
    # Compile all features into a single vector
    features = np.concatenate([
        aggregated_features.flatten(),
        aggregated_features2.flatten(),
        range_max_min.flatten(),
        # kurt.flatten(),
        delta_mean.flatten(),
        delta_std.flatten(),
        delta_max.flatten(),
        delta_min.flatten(),
        delta_mean2.flatten(),
        delta_std2.flatten(),
        delta_max2.flatten(),
        delta_min2.flatten(),
        energy_entropy.flatten(),
        energy_entropy_delta.flatten(),
        energy_entropy_delta2.flatten(),
        # mfcc_rolling_mean.values.flatten(),
        # mfcc_rolling_std.values.flatten(),
        # mfcc_correlation.flatten(),
    ])

    # print(f'Processed {file_name}')
        
    generated_features.append(features)  # Append the features for this song
    file_names.append(file_name)          # Store the file name

# Stack the generated features into a numpy array
min_length = min(features.shape[0] for features in generated_features)
truncated_features = [features[:min_length] for features in generated_features]
generated_features = np.vstack(truncated_features)
print(f'\nGenerated {generated_features.shape[0]} feature vectors with {generated_features.shape[1]} features each')
total_features = generated_features.shape[1]
feature_columns = [f'feature_{i}' for i in range(total_features)]

# Create a DataFrame to store the features and corresponding file names
features_df = pd.DataFrame(generated_features, columns=feature_columns)
features_df.insert(0, 'File', file_names)

# Save generated features to 'features_generated_for_clustering.csv'
features_df.to_csv(f'features_generated_known_data.csv', index=False)
