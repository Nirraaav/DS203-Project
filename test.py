import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Function to load and aggregate MFCC data based only selective coefficients 
def aggregate_mfcc_selective(mfcc_data):
    """
    Aggregates MFCC data by selecting specific components
    and then calculating the mean, standard deviation, max, and min for these selected coefficients.
    """
    mfcc_selected = mfcc_data[13:, :]  # Slice rows for MFCC 
    
    # Aggregate the selected MFCCs
    mfcc_mean = np.mean(mfcc_selected, axis=1)  # Mean across time
    mfcc_std = np.std(mfcc_selected, axis=1)    # Standard deviation across time
    mfcc_max = np.max(mfcc_selected, axis=1)    # Max value across time
    mfcc_min = np.min(mfcc_selected, axis=1)    # Min value across time
    
    # Concatenate the aggregated features into one feature vector
    features = np.concatenate([mfcc_mean, mfcc_std, mfcc_max, mfcc_min])
    return features

# Load MFCC data from CSV files and aggregate them based on MFCC 2-5
mfcc_all_songs = []
file_names = []
for i in range(1, 117):  # Assuming 115 songs are numbered as '01-MFCC.csv', '02-MFCC.csv', ..., '115-MFCC.csv'
    # Load the MFCC CSV file for each song
    file_name = f'data/{i:02d}-MFCC.csv'
    song_data = pd.read_csv(file_name, header=None)
    # Aggregate only MFCC 2 to 5
    aggregated_features = aggregate_mfcc_selective(song_data.values)
    mfcc_all_songs.append(aggregated_features)
    file_names.append(file_name)  # Store file names

# Convert the list of all songs' aggregated MFCC features into a NumPy array
mfcc_all_songs = np.array(mfcc_all_songs)

# Standardize the features (zero mean, unit variance)
scaler = StandardScaler()
mfcc_scaled = scaler.fit_transform(mfcc_all_songs)

# Perform K-Means clustering
k = 6  # Set the number of clusters (you can try different values for k)
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(mfcc_scaled)

# Get the cluster labels for each song
cluster_labels = kmeans.labels_

# Evaluate the clustering with different metrics
silhouette_avg = silhouette_score(mfcc_scaled, cluster_labels)
db_index = davies_bouldin_score(mfcc_scaled, cluster_labels)
ch_score = calinski_harabasz_score(mfcc_scaled, cluster_labels)

print(f'Silhouette Score: {silhouette_avg:.4f}')
print(f'Davies-Bouldin Index: {db_index:.4f}')
print(f'Calinski-Harabasz Score: {ch_score:.4f}')

# Output the cluster label for each file
print("\nCluster Labels for Each File (Based on MFCC 2-5):")
for file_name, label in zip(file_names, cluster_labels):
    print(f'{file_name}: Cluster {label}')

# Visualize the clusters using PCA to reduce the data to 2D
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(mfcc_scaled)

# Plot the PCA-reduced data, colored by cluster
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis', s=50)
plt.title('K-Means Clustering (Based on MFCC 2-5) - PCA-reduced 2D projection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

# Save the file names and cluster labels to a CSV file
output_df = pd.DataFrame({
    'File': file_names,
    'Cluster': cluster_labels
})
output_df.to_csv('file_cluster_labels.csv', index=False)
    