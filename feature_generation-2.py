import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import librosa
import matplotlib.pyplot as plt
from librosa.feature.rhythm import tempo as compute_tempo_function
from scipy.stats import skew, kurtosis

INDEX = 13

def aggregate_mfcc_selective(mfcc_data):
    mfcc_selected = mfcc_data[INDEX:, :] 
    
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
data_directory = 'data-v2'
files = [file for file in os.listdir(data_directory) if file.endswith("-MFCC.csv")]
files.sort()  # Sort the files alphabetically

for file_name in files:
    file_path = os.path.join(data_directory, file_name)
    mfcc_data = pd.read_csv(file_path, header=None).values

    # Compute aggregated MFCC features
    aggregated_features = aggregate_mfcc_selective(mfcc_data)
    skewness = skew(mfcc_data, axis=1)
    kurt = kurtosis(mfcc_data, axis=1)
    range_max_min = np.ptp(mfcc_data, axis=1)

    delta_mfcc = np.diff(mfcc_data, axis=1)
    delta_delta_mfcc = np.diff(delta_mfcc, axis=1)

    delta_mean = np.mean(delta_mfcc, axis=1)
    delta_std = np.std(delta_mfcc, axis=1)
    delta_max = np.max(delta_mfcc, axis=1)
    delta_min = np.min(delta_mfcc, axis=1)
        
    # Compile all features into a single vector
    features = np.concatenate([
        aggregated_features.flatten(),
        range_max_min.flatten(),
        delta_mean.flatten(),
        delta_std.flatten(),
        delta_max.flatten(),
        delta_min.flatten(),
    ])

    print(f'Processed {file_name}')
        
    generated_features.append(features)  # Append the features for this song
    file_names.append(file_name)          # Store the file name

# Stack the generated features into a numpy array
generated_features = np.vstack(generated_features)
total_features = generated_features.shape[1]
feature_columns = [f'feature_{i}' for i in range(total_features)]

# Create a DataFrame to store the features and corresponding file names
features_df = pd.DataFrame(generated_features, columns=feature_columns)
features_df.insert(0, 'File', file_names)

# Save generated features to 'features_generated.csv'
features_df.to_csv(f'features_generated_{INDEX}.csv', index=False)

# Standardize the features before PCA
scaler = StandardScaler()
mfcc_scaled = scaler.fit_transform(generated_features)

# Apply PCA on the entire feature set (instead of on raw MFCC data)
pca = PCA(n_components=5)  # Adjust number of components if needed
pca_features = pca.fit_transform(mfcc_scaled)

# Perform K-Means clustering on PCA-reduced data
k = 6  # Number of clusters (you can experiment with different values)
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(pca_features)

# Get cluster labels for each song
cluster_labels = kmeans.labels_

# Evaluate clustering with different metrics
silhouette_avg = silhouette_score(pca_features, cluster_labels)
db_index = davies_bouldin_score(pca_features, cluster_labels)
ch_score = calinski_harabasz_score(pca_features, cluster_labels)

print(f'Silhouette Score: {silhouette_avg:.4f}')
print(f'Davies-Bouldin Index: {db_index:.4f}')
print(f'Calinski-Harabasz Score: {ch_score:.4f}')

# Visualize clusters using PCA for 2D projection (already PCA-reduced)
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=cluster_labels, cmap='viridis', s=50)
plt.title('K-Means Clustering on PCA-Reduced Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.savefig('cluster_plot.png')
plt.show()

# Assign descriptive cluster labels
cluster_label_mapping = {
    0: "Other Artist 1",
    1: "Kishore Kumar",
    2: "Michael Jackson",
    3: "Jana Gana Mana",
    4: "Asha Bhosale",  # Replace with actual song/artist name if known
    5: "Other Artist 2"   # Replace with actual song/artist name if known
}

# Save file names and cluster labels to 'file_cluster_labels.csv'
output_df = pd.DataFrame({
    'File': file_names,
    'Cluster': cluster_labels
})

output_df['Cluster'] = output_df['Cluster'].map(cluster_label_mapping)

print("\nDescriptive Cluster Labels for Each File:")
for file_name, label in zip(file_names, output_df['Cluster']):
    print(f'{file_name}: {label}')

# Save the file names and corresponding clusters with descriptive labels
output_df.to_csv(f'labels/file_cluster_labels_{INDEX}.csv', index=False)

