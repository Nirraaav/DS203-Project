import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from collections import Counter
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load or generate features if needed
INDEX = 13
features_file = f'extracted_features.csv'
if os.path.exists(features_file):
    features_df = pd.read_csv(features_file)
    generated_features = features_df.iloc[:, 1:].values
    file_names = features_df['song_number'].tolist()
else:
    print("Feature file not found. Please generate features first.")
    exit()

# Standardize the features
scaler = StandardScaler()
mfcc_scaled = scaler.fit_transform(generated_features)

# Apply PCA for distance-based clustering algorithms
# pca = PCA(n_components=5)
# pca_data = pca.fit_transform(mfcc_scaled)

pca_data = mfcc_scaled

# Define number of clusters
clusters = 6

# Define clustering algorithms
clustering_algorithms = {
    'KMeans': KMeans(n_clusters=clusters, random_state=42),
    'AgglomerativeClustering': AgglomerativeClustering(n_clusters=clusters),
    'Birch': Birch(n_clusters=clusters),
    'SpectralClustering': SpectralClustering(n_clusters=clusters, affinity='nearest_neighbors', n_neighbors=10, assign_labels='discretize', random_state=42),
    'GaussianMixture': GaussianMixture(n_components=clusters, random_state=42)
}

# Dictionary to store cluster labels for each model
cluster_labels = {}

# Apply each clustering algorithm
for name, algorithm in clustering_algorithms.items():
    if name == 'SpectralClustering':
        # SpectralClustering needs original scaled data, not PCA
        labels = algorithm.fit_predict(mfcc_scaled)
    else:
        labels = algorithm.fit_predict(pca_data)
    
    cluster_labels[name] = labels
    print(f"{name} clustering completed.")

# Evaluate clustering metrics
print("\nEvaluation Metrics for Clustering:")

for name, labels in cluster_labels.items():
    silhouette = silhouette_score(mfcc_scaled, labels)
    davies_bouldin = davies_bouldin_score(mfcc_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(mfcc_scaled, labels)
    print(f"{name} - Silhouette: {silhouette},\nDavies-Bouldin: {davies_bouldin},\nCalinski-Harabasz: {calinski_harabasz}\n\n")

# Save clustering results in a DataFrame
cluster_labels_df = pd.DataFrame({
    'File': file_names,
    'KMeans': cluster_labels['KMeans'],
    'Agglomerative': cluster_labels['AgglomerativeClustering'],
    'Birch': cluster_labels['Birch'],
    'SpectralClustering': cluster_labels['SpectralClustering'],
    'GaussianMixture': cluster_labels['GaussianMixture']
})
output_file = f'labels/song_cluster_labels_{INDEX}.csv'
cluster_labels_df.to_csv(output_file, index=False)
print(f"Saved clustering labels to {output_file}")

# -------- VIF Calculation --------
# Function to calculate VIF
def calculate_vif(features):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = [f'Feature_{i}' for i in range(features.shape[1])]
    vif_data["VIF"] = [variance_inflation_factor(features, i) for i in range(features.shape[1])]
    return vif_data

# Apply VIF calculation to the scaled MFCC features
vif_df = calculate_vif(mfcc_scaled)

# Print the VIF results
print("\nVariance Inflation Factor (VIF) for Features:")
print(vif_df)

# VIF Bar Graphs
plt.figure(figsize=(10, 5))
plt.bar(vif_df['Feature'], vif_df['VIF'])
plt.title('Variance Inflation Factor (VIF) for Features')
plt.xlabel('Feature')
plt.ylabel('VIF')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Optionally, filter out columns with high VIF (e.g., VIF > 5 or 10)
high_vif_threshold = 10**6  # You can adjust this threshold based on your criteria
high_vif_features = vif_df[vif_df['VIF'] > high_vif_threshold]
print(f"\nFeatures with high VIF (> {high_vif_threshold}):")
print(high_vif_features)

# delete high VIF features
filtered_features = np.delete(mfcc_scaled, high_vif_features.index, axis=1)
print(f"\nFiltered features shape: {filtered_features.shape}")

# Visualizations of clustering results

# run PCA on filtered features
pca_filtered = PCA(n_components=5)
pca_filtered_data = pca_filtered.fit_transform(filtered_features)

# cluster the filtered features
# Apply each clustering algorithm
for name, algorithm in clustering_algorithms.items():
    if name == 'SpectralClustering':
        # SpectralClustering needs original scaled data, not PCA
        labels = algorithm.fit_predict(mfcc_scaled)
    else:
        labels = algorithm.fit_predict(pca_data)
    
    cluster_labels[name] = labels
    print(f"{name} clustering completed.")

# Evaluate clustering metrics
print("\nEvaluation Metrics for Clustering:")

for name, labels in cluster_labels.items():
    silhouette = silhouette_score(mfcc_scaled, labels)
    davies_bouldin = davies_bouldin_score(mfcc_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(mfcc_scaled, labels)
    print(f"{name} - Silhouette: {silhouette},\nDavies-Bouldin: {davies_bouldin},\nCalinski-Harabasz: {calinski_harabasz}\n\n")
