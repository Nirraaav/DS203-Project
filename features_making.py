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
import plotly.express as px
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch, MeanShift, SpectralClustering, OPTICS, AffinityPropagation
from sklearn.mixture import GaussianMixture

INDEX = 1  # Start index for MFCC features you want to include

# Dictionary to store clustering results
results = {}

# Define clustering algorithms with their default settings
clustering_algorithms = {
    'KMeans': KMeans(n_clusters=3, random_state=42),
    'AgglomerativeClustering': AgglomerativeClustering(n_clusters=3),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
    'Birch': Birch(n_clusters=3),
    'MeanShift': MeanShift(),
    'SpectralClustering': SpectralClustering(n_clusters=3, random_state=42),
    'OPTICS': OPTICS(min_samples=5),
    'AffinityPropagation': AffinityPropagation(random_state=42),
    'GaussianMixture': GaussianMixture(n_components=3, random_state=42)
}

def extract_features(mfcc_data):
    # Select the MFCC data starting from the specified INDEX
    mfcc_selected = mfcc_data[INDEX:, :]
    mfcc_flattened = mfcc_selected.flatten()
    # Calculate features for each MFCC coefficient
    features = {
        'mean': np.mean(mfcc_selected, axis=1),
        'std': np.std(mfcc_selected, axis=1),
        #'max': np.max(mfcc_selected, axis=1),
        #'min': np.min(mfcc_selected, axis=1),
        'delta': np.mean(librosa.feature.delta(mfcc_selected), axis=1),
        'delta2': np.mean(librosa.feature.delta(mfcc_selected, order=2), axis=1),
        #'delta_max': np.max(librosa.feature.delta(mfcc_selected), axis=1),
        #'delta2_max': np.max(librosa.feature.delta(mfcc_selected, order=2), axis=1),
        #'delta_min': np.min(librosa.feature.delta(mfcc_selected), axis=1),
        #'delta2_min': np.min(librosa.feature.delta(mfcc_selected, order=2), axis=1),
        'skew': skew(mfcc_selected, axis=1),
        'kurtosis': kurtosis(mfcc_selected, axis=1),
        'range': np.ptp(mfcc_selected, axis=1),
        #'total_energy': np.sum(mfcc_selected ** 2, axis=1),
        'energy_entropy': -np.sum(mfcc_selected ** 2 * np.log(mfcc_selected ** 2 + 1e-10), axis=1),
        #'geometric_mean': np.exp(np.mean(np.log(np.abs(mfcc_selected[mfcc_selected > 0]) + 1e-6))), 
        #'arithmetic_mean': np.nanmean(np.abs(mfcc_selected)),
        #'mfcc_spectral_flatness': np.exp(np.mean(np.log(np.abs(mfcc_selected) + 1e-6))) / (np.mean(np.abs(mfcc_selected)) + 1e-10)
    }
    return features

# Function to evaluate clustering
def evaluate_clustering(labels, scaled_data):
    scores = {}
    if len(set(labels)) > 1:  # Avoid errors when only one cluster is formed
        scores['Silhouette Score'] = silhouette_score(scaled_data, labels)
        scores['Davies-Bouldin Index'] = davies_bouldin_score(scaled_data, labels)
        scores['Calinski-Harabasz Score'] = calinski_harabasz_score(scaled_data, labels)
    else:
        scores['Silhouette Score'] = 'N/A'
        scores['Davies-Bouldin Index'] = 'N/A'
        scores['Calinski-Harabasz Score'] = 'N/A'
    return scores

# Define DataFrame columns
columns = []
for stat in [ 'mean', 'std', 'delta', 'delta2', 'skew', 'kurtosis', 'range', 'energy_entropy']:
    columns.extend([f'{stat}_mfcc_{i}' for i in range(INDEX, 20)])  9

# Initialize an empty DataFrame to store all features
all_features_df = pd.DataFrame(columns=['song_number'] + columns)

for i in range(1, 117):
    file_name = f'data-v2/{i:02d}-MFCC.csv'
    mfcc_data = pd.read_csv(file_name, header=None).values

    # Extract features
    features = extract_features(mfcc_data)
    
    # Flatten the feature dictionary into a list
    row = [f'song_{i}']
    for stat in features:
        if hasattr(features[stat], '__len__'):
            row.extend(features[stat][j] for j in range(len(features[stat])))
        else:
            row.append(features[stat])
    
    # Ensure the row length matches the DataFrame column count
    if len(row) == len(all_features_df.columns):
        all_features_df.loc[i-1] = row
    else:
        print(f"Row length mismatch for song {i}: expected {len(all_features_df.columns)}, got {len(row)}")


# Generate the correlation matrix
heatmap_data = all_features_df.drop(columns=['song_number'])
correlation_matrix = heatmap_data.corr()

scaler = StandardScaler()
mfcc_features_scaled = scaler.fit_transform(heatmap_data)
mfcc_features_scaled_df = pd.DataFrame(mfcc_features_scaled, columns=heatmap_data.columns)

# Create a heatmap using Plotly
fig = px.imshow(correlation_matrix,
                color_continuous_scale='RdBu',
                title='Heatmap of Feature Correlations',
                labels=dict(x="Features", y="Features", color="Correlation"),
                aspect="auto")

# Update layout for better visibility
fig.update_layout(xaxis_title='Features', yaxis_title='Features', title_x=0.5)

# Show the heatmap
fig.show()

pca = PCA(n_components=5)
pca_components = pca.fit_transform(mfcc_features_scaled_df)

pca_df = pd.DataFrame(pca_components, columns=[f'PCA_{i+1}' for i in range(pca_components.shape[1])])
mfcc_scaled_with_pca = pd.concat([mfcc_features_scaled_df, pca_df], axis=1)

# # Run each clustering algorithm and store results
# for name, algorithm in clustering_algorithms.items():
#     try:
#         if name == 'GaussianMixture':
#             labels = algorithm.fit_predict(mfcc_scaled_with_pca)
#         else:
#             labels = algorithm.fit(mfcc_scaled_with_pca).labels_
        
#         # Evaluate clustering
#         results[name] = evaluate_clustering(labels,mfcc_scaled_with_pca)
#     except Exception as e:
#         results[name] = f"Failed with error: {e}"

# # Display results
# for algo, scores in results.items():
#     print(f"\nAlgorithm: {algo}")
#     if isinstance(scores, dict):
#         for metric, score in scores.items():
#             print(f"{metric}: {score}")
#     else:
#         print(scores)

# Perform K-Means clustering
k = 3  # Number of clusters (you can experiment with different values)
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(mfcc_scaled_with_pca.iloc[:, 1:])  # Fit the model to all features except the song number

cluster_labels = kmeans.labels_
all_features_df.insert(1, 'cluster_name', cluster_labels)

# Evaluate clustering with different metrics
silhouette_avg = silhouette_score(mfcc_features_scaled_df, cluster_labels)
db_index = davies_bouldin_score(mfcc_features_scaled_df, cluster_labels)
ch_score = calinski_harabasz_score(mfcc_features_scaled_df, cluster_labels)

print(f'Silhouette Score: {silhouette_avg:.4f}')
print(f'Davies-Bouldin Index: {db_index:.4f}')
print(f'Calinski-Harabasz Score: {ch_score:.4f}')

# Save the DataFrame to a CSV file
all_features_df.to_csv('extracted_features.csv', index=False)
print("Features saved to extracted_features.csv")



# Plot PCA-reduced data, colored by cluster
#plt.scatter(pca_components[:, 0], pca_components[:, 1], c=cluster_labels, cmap='viridis', s=50)
#plt.title('K-Means Clustering on Generated Features (PCA-reduced 2D projection)')
#plt.xlabel('Principal Component 1')
#plt.ylabel('Principal Component 2')
#plt.colorbar(label='Cluster Label')
#plt.show()

