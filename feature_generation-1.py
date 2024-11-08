import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter
import os

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
pca = PCA(n_components=10)
pca_data = pca.fit_transform(mfcc_scaled)

# Apply t-SNE for density-based clustering algorithms
# tsne = TSNE(n_components=3, perplexity=30, random_state=42)
# tsne_data = tsne.fit_transform(mfcc_scaled)

# Clustering with KMeans
k = 6
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans_labels = kmeans.fit_predict(pca_data)

# Clustering with Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=k)
agg_labels = agg_clustering.fit_predict(pca_data)

# Clustering with OPTICS (density-based clustering on t-SNE-reduced data)
optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1)
optics_labels = optics.fit_predict(tsne_data)

# Evaluate clustering metrics
print("\nEvaluation Metrics for Clustering:")

# Metrics for KMeans
silhouette_kmeans = silhouette_score(mfcc_scaled, kmeans_labels)
davies_bouldin_kmeans = davies_bouldin_score(mfcc_scaled, kmeans_labels)
calinski_harabasz_kmeans = calinski_harabasz_score(mfcc_scaled, kmeans_labels)
print(f"KMeans - Silhouette: {silhouette_kmeans}, Davies-Bouldin: {davies_bouldin_kmeans}, Calinski-Harabasz: {calinski_harabasz_kmeans}")

# Metrics for Agglomerative Clustering
silhouette_agg = silhouette_score(mfcc_scaled, agg_labels)
davies_bouldin_agg = davies_bouldin_score(mfcc_scaled, agg_labels)
calinski_harabasz_agg = calinski_harabasz_score(mfcc_scaled, agg_labels)
print(f"Agglomerative - Silhouette: {silhouette_agg}, Davies-Bouldin: {davies_bouldin_agg}, Calinski-Harabasz: {calinski_harabasz_agg}")

# Metrics for OPTICS, if clusters exist
if len(set(optics_labels)) > 1:
    silhouette_optics = silhouette_score(mfcc_scaled, optics_labels)
    davies_bouldin_optics = davies_bouldin_score(mfcc_scaled, optics_labels)
    calinski_harabasz_optics = calinski_harabasz_score(mfcc_scaled, optics_labels)
    print(f"OPTICS - Silhouette: {silhouette_optics}, Davies-Bouldin: {davies_bouldin_optics}, Calinski-Harabasz: {calinski_harabasz_optics}")
else:
    print("OPTICS did not produce enough clusters to compute metrics.")

# Save clustering results in a DataFrame
cluster_labels_df = pd.DataFrame({
    'File': file_names,
    'KMeans': kmeans_labels,
    'Agglomerative': agg_labels,
    'OPTICS': optics_labels
})
output_file = f'labels/song_cluster_labels_{INDEX}.csv'
cluster_labels_df.to_csv(output_file, index=False)
print(f"Saved clustering labels to {output_file}")

# Visualizations

# Visualize KMeans and Agglomerative Clustering on PCA-reduced data
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans_labels, cmap='viridis', s=50)
plt.title('KMeans Clustering (PCA-reduced)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.subplot(1, 2, 2)
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=agg_labels, cmap='viridis', s=50)
plt.title('Agglomerative Clustering (PCA-reduced)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.tight_layout()
plt.savefig('pca_distance_based_clustering.png')
plt.show()

# Visualize OPTICS clustering on t-SNE-reduced data
plt.figure(figsize=(6, 5))
plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=optics_labels, cmap='plasma', s=50)
plt.title('OPTICS Clustering (t-SNE-reduced)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Cluster Label')
plt.savefig('tsne_density_based_clustering.png')
plt.show()
