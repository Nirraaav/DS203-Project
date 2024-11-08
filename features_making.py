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
from sklearn.metrics import pairwise_distances
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

INDEX = 1  # Start index for MFCC features you want to include

# Dictionary to store clustering results
results = {}

# Define clustering algorithms with their default settings
clustering_algorithms = {
    'KMeans': KMeans(n_clusters=3, random_state=42),
    'AgglomerativeClustering': AgglomerativeClustering(),
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
    print(mfcc_selected.shape)
    # Calculate features for each MFCC coefficient
    features = {
        'mean': np.mean(mfcc_selected, axis=1),
        'std': np.std(mfcc_selected, axis=1),
        'max': np.max(mfcc_selected, axis=1),
        'min': np.min(mfcc_selected, axis=1),
        'skew': skew(mfcc_selected, axis=1),
        'kurtosis': kurtosis(mfcc_selected, axis=1),
        'range': np.ptp(mfcc_selected, axis=1),
        'total_energy': np.sum(mfcc_selected ** 2, axis=1),
        'energy_entropy': -np.sum(mfcc_selected ** 2 * np.log(mfcc_selected ** 2 + 1e-10), axis=1),

        'delta_mean': np.mean(librosa.feature.delta(mfcc_selected), axis=1),
        'delta_std': np.std(librosa.feature.delta(mfcc_selected), axis=1),
        'delta_max': np.max(librosa.feature.delta(mfcc_selected), axis=1),
        'delta_min': np.min(librosa.feature.delta(mfcc_selected), axis=1),
        'delta_skew': skew(librosa.feature.delta(mfcc_selected), axis=1),
        'delta_kurtosis': kurtosis(librosa.feature.delta(mfcc_selected), axis=1),
        'delta_range': np.ptp(librosa.feature.delta(mfcc_selected), axis=1),    
        'delta_total_energy': np.sum(librosa.feature.delta(mfcc_selected) ** 2, axis=1),
        'delta_energy_entropy': -np.sum(librosa.feature.delta(mfcc_selected) ** 2 * np.log(librosa.feature.delta(mfcc_selected) ** 2 + 1e-10), axis=1),

        'delta2_mean': np.mean(librosa.feature.delta(mfcc_selected, order=2), axis=1),
        'delta2_std': np.std(librosa.feature.delta(mfcc_selected, order=2), axis=1),
        'delta2_max': np.max(librosa.feature.delta(mfcc_selected, order=2), axis=1),
        'delta2_min': np.min(librosa.feature.delta(mfcc_selected, order=2), axis=1),
        'delta2_skew': skew(librosa.feature.delta(mfcc_selected, order=2), axis=1),
        'delta2_kurtosis': kurtosis(librosa.feature.delta(mfcc_selected, order=2), axis=1),
        'delta2_range': np.ptp(librosa.feature.delta(mfcc_selected, order=2), axis=1),  
        'delta2_total_energy': np.sum(librosa.feature.delta(mfcc_selected, order=2) ** 2, axis=1),
        'delta2_energy_entropy': -np.sum(librosa.feature.delta(mfcc_selected, order=2) ** 2 * np.log(librosa.feature.delta(mfcc_selected, order=2) ** 2 + 1e-10), axis=1),
    }
    features2 = []
    for i in range(len(mfcc_selected)):
        features_temp = []
        features_temp.append(np.mean(np.correlate(mfcc_selected[i], mfcc_selected[i], mode='full')[len(mfcc_selected[i])-1:]))
        features_temp.append(np.std(np.correlate(mfcc_selected[i], mfcc_selected[i], mode='full')[len(mfcc_selected[i])-1:]))
        features_temp.append(np.max(np.correlate(mfcc_selected[i], mfcc_selected[i], mode='full')[len(mfcc_selected[i])-1:]))
        features_temp.append(np.min(np.correlate(mfcc_selected[i], mfcc_selected[i], mode='full')[len(mfcc_selected[i])-1:]))
        features_temp.append(skew(np.correlate(mfcc_selected[i], mfcc_selected[i], mode='full')[len(mfcc_selected[i])-1:]))
        features_temp.append(kurtosis(np.correlate(mfcc_selected[i], mfcc_selected[i], mode='full')[len(mfcc_selected[i])-1:]))
        features_temp.append(np.ptp(np.correlate(mfcc_selected[i], mfcc_selected[i], mode='full')[len(mfcc_selected[i])-1:]))
        features_temp.append(np.sum(np.correlate(mfcc_selected[i], mfcc_selected[i], mode='full')[len(mfcc_selected[i])-1:] ** 2))
        features_temp.append(-np.sum(np.correlate(mfcc_selected[i], mfcc_selected[i], mode='full')[len(mfcc_selected[i])-1:] ** 2 * np.log(np.correlate(mfcc_selected[i], mfcc_selected[i], mode='full')[len(mfcc_selected[i])-1:] ** 2 + 1e-10)))

        features_temp.append(np.mean(np.correlate(mfcc_selected[i], mfcc_selected[i][::-1], mode='full')[len(mfcc_selected[i])-1:]))
        features_temp.append(np.std(np.correlate(mfcc_selected[i], mfcc_selected[i][::-1], mode='full')[len(mfcc_selected[i])-1:]))
        features_temp.append(np.max(np.correlate(mfcc_selected[i], mfcc_selected[i][::-1], mode='full')[len(mfcc_selected[i])-1:]))
        features_temp.append(np.min(np.correlate(mfcc_selected[i], mfcc_selected[i][::-1], mode='full')[len(mfcc_selected[i])-1:]))
        features_temp.append(skew(np.correlate(mfcc_selected[i], mfcc_selected[i][::-1], mode='full')[len(mfcc_selected[i])-1:]))
        features_temp.append(kurtosis(np.correlate(mfcc_selected[i], mfcc_selected[i][::-1], mode='full')[len(mfcc_selected[i])-1:]))
        features_temp.append(np.ptp(np.correlate(mfcc_selected[i], mfcc_selected[i][::-1], mode='full')[len(mfcc_selected[i])-1:]))
        features_temp.append(np.sum(np.correlate(mfcc_selected[i], mfcc_selected[i][::-1], mode='full')[len(mfcc_selected[i])-1:] ** 2))
        features_temp.append(-np.sum(np.correlate(mfcc_selected[i], mfcc_selected[i][::-1], mode='full')[len(mfcc_selected[i])-1:] ** 2 * np.log(np.correlate(mfcc_selected[i], mfcc_selected[i][::-1], mode='full')[len(mfcc_selected[i])-1:] ** 2 + 1e-10)))

        features_temp.append(np.mean(np.convolve(mfcc_selected[i], mfcc_selected[i], mode='full')))
        features_temp.append(np.std(np.convolve(mfcc_selected[i], mfcc_selected[i], mode='full')))
        features_temp.append(np.max(np.convolve(mfcc_selected[i], mfcc_selected[i], mode='full')))
        features_temp.append(np.min(np.convolve(mfcc_selected[i], mfcc_selected[i], mode='full')))
        features_temp.append(skew(np.convolve(mfcc_selected[i], mfcc_selected[i], mode='full')))
        features_temp.append(kurtosis(np.convolve(mfcc_selected[i], mfcc_selected[i], mode='full')))
        features_temp.append(np.ptp(np.convolve(mfcc_selected[i], mfcc_selected[i], mode='full')))
        features_temp.append(np.sum(np.convolve(mfcc_selected[i], mfcc_selected[i], mode='full') ** 2))
        features_temp.append(-np.sum(np.convolve(mfcc_selected[i], mfcc_selected[i], mode='full') ** 2 * np.log(np.convolve(mfcc_selected[i], mfcc_selected[i], mode='full') ** 2 + 1e-10)))

        features_temp.append(np.mean(np.convolve(mfcc_selected[i], mfcc_selected[i][::-1], mode='full')))
        features_temp.append(np.std(np.convolve(mfcc_selected[i], mfcc_selected[i][::-1], mode='full')))
        features_temp.append(np.max(np.convolve(mfcc_selected[i], mfcc_selected[i][::-1], mode='full')))
        features_temp.append(np.min(np.convolve(mfcc_selected[i], mfcc_selected[i][::-1], mode='full')))
        features_temp.append(skew(np.convolve(mfcc_selected[i], mfcc_selected[i][::-1], mode='full')))
        features_temp.append(kurtosis(np.convolve(mfcc_selected[i], mfcc_selected[i][::-1], mode='full')))
        features_temp.append(np.ptp(np.convolve(mfcc_selected[i], mfcc_selected[i][::-1], mode='full')))
        features_temp.append(np.sum(np.convolve(mfcc_selected[i], mfcc_selected[i][::-1], mode='full') ** 2))
        features_temp.append(-np.sum(np.convolve(mfcc_selected[i], mfcc_selected[i][::-1], mode='full') ** 2 * np.log(np.convolve(mfcc_selected[i], mfcc_selected[i][::-1], mode='full') ** 2 + 1e-10)))

        features2.append(features_temp)

    feature_names = ['autocorrelation_mean', 'autocorrelation_std', 'autocorrelation_max', 'autocorrelation_min', 'autocorrelation_skew', 'autocorrelation_kurtosis', 'autocorrelation_range', 'autocorrelation_total_energy', 'autocorrelation_energy_entropy', 'cross_correlation_mean', 'cross_correlation_std', 'cross_correlation_max', 'cross_correlation_min', 'cross_correlation_skew', 'cross_correlation_kurtosis', 'cross_correlation_range', 'cross_correlation_total_energy', 'cross_correlation_energy_entropy', 'convolution_mean', 'convolution_std', 'convolution_max', 'convolution_min', 'convolution_skew', 'convolution_kurtosis', 'convolution_range', 'convolution_total_energy', 'convolution_energy_entropy', 'cross_convolution_mean', 'cross_convolution_std', 'cross_convolution_max', 'cross_convolution_min', 'cross_convolution_skew', 'cross_convolution_kurtosis', 'cross_convolution_range', 'cross_convolution_total_energy', 'cross_convolution_energy_entropy']

    for i in range(len(feature_names)):
        features[feature_names[i]] = [features2[j][i] for j in range(len(features2))]

    # for i in range(len(features)):
    #     print(f'{list(features.keys())[i]}: {len(list(features.values())[i])}')
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
for stat in ['mean', 'std', 'max', 'min', 'skew', 'kurtosis', 'range', 'total_energy', 'energy_entropy', 'delta_mean', 'delta_std', 'delta_max', 
             'delta_min', 'delta_skew', 'delta_kurtosis', 'delta_range', 'delta_total_energy', 'delta_energy_entropy', 'delta2_mean', 'delta_std', 
             'delta2_max', 'delta2_min', 'delta2_skew', 'delta2_kurtosis', 'delta2_range', 'delta2_total_energy', 'delta2_energy_entropy', 
             'autocorrelation_mean', 'autocorrelation_std', 'autocorrelation_max', 'autocorrelation_min', 'autocorrelation_skew', 'autocorrelation_kurtosis',
               'autocorrelation_range', 'autocorrelation_total_energy', 'autocorrelation_energy_entropy', 'cross_correlation_mean', 'cross_correlation_std', 
               'cross_correlation_max', 'cross_correlation_min', 'cross_correlation_skew', 'cross_correlation_kurtosis', 'cross_correlation_range', 
               'cross_correlation_total_energy', 'cross_correlation_energy_entropy', 'convolution_mean', 'convolution_std', 'convolution_max', 
               'convolution_min', 'convolution_skew', 'convolution_kurtosis', 'convolution_range', 'convolution_total_energy', 'convolution_energy_entropy', 
               'cross_convolution_mean', 'cross_convolution_std', 'cross_convolution_max', 'cross_convolution_min', 'cross_convolution_skew', 
               'cross_convolution_kurtosis', 'cross_convolution_range', 'cross_convolution_total_energy', 'cross_convolution_energy_entropy']:
    columns.extend([f'{stat}_mfcc_{i}' for i in range(INDEX, 20)])  

# Initialize an empty DataFrame to store all features
all_features_df = pd.DataFrame(columns=['song_number'] + columns)


for i in range(1, 117):
    file_name = f'data-v2/{i:02d}-MFCC.csv'
    mfcc_data = pd.read_csv(file_name, header=None).values
    # Create subplots - grid with 5 rows and 4 columns (can adjust as needed)
    # fig = make_subplots(
    #     rows=5, cols=4,  # 5 rows, 4 columns (you can adjust the grid size as needed)
    #     subplot_titles=[f'MFCC Feature {i+1}' for i in range(20)]  # Titles for each subplot
    # )

    # # Add scatter plot for each MFCC feature
    # for i in range(20):
    #     row = i // 4 + 1  # Row index (5 rows)
    #     col = i % 4 + 1   # Column index (4 columns)
        
    #     fig.add_trace(
    #         go.Scatter(
    #             x=list(range(mfcc_data.shape[1])),  # x = time frame index
    #             y=mfcc_data[i, :],  # y = MFCC feature values
    #             mode='markers',
    #             name=f'MFCC Feature {i+1}'  # Label for the feature
    #         ),
    #         row=row, col=col
    #     )

    # # Update layout for better visualization
    # fig.update_layout(
    #     title='MFCC Features Grid of Scatter Plots',
    #     height=1000,  # Adjust height for better visibility
    #     showlegend=False,  # Optional: Hide legend for clarity
    #     template='plotly_dark',  # Optional theme
    # )

    # # Show the plot
    # fig.show()
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
feature_data = all_features_df.drop(columns=['song_number'])
correlation_matrix = feature_data.corr()

scaler = StandardScaler()
feature_data_scaled = scaler.fit_transform(feature_data)
feature_data_scaled_df = pd.DataFrame(feature_data_scaled, columns=feature_data.columns)

# Calculate Euclidean distances between each pair of songs
distance_matrix = pairwise_distances(feature_data_scaled.T, metric='euclidean')

# Create a DataFrame for the distance matrix for easy visualization
distance_df = pd.DataFrame(distance_matrix, index=feature_data.columns, columns=feature_data.columns)

# Create a heatmap using Plotly
fig = px.imshow(distance_df,
                color_continuous_scale='Viridis',
                title='Euclidean Distance Heatmap between Songs',
                labels=dict(x="Song", y="Features", color="Euclidean Distance"),
                aspect="auto")

# Update layout for better visibility
fig.update_layout(xaxis_title='Songs', yaxis_title='Features', title_x=0.5)

# Show the heatmap
fig.show()



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

pca = PCA(n_components=80)
pca_components = pca.fit_transform(feature_data_scaled_df)

pca_df = pd.DataFrame(pca_components, columns=[f'PCA_{i+1}' for i in range(pca_components.shape[1])])
mfcc_scaled_with_pca = pd.concat([feature_data_scaled_df, pca_df], axis=1)

# Run each clustering algorithm and store results
for name, algorithm in clustering_algorithms.items():
    try:
        if name == 'GaussianMixture':
            labels = algorithm.fit_predict(mfcc_scaled_with_pca)
        else:
            labels = algorithm.fit(mfcc_scaled_with_pca).labels_
        
        # Evaluate clustering
        results[name] = evaluate_clustering(labels,mfcc_scaled_with_pca)
    except Exception as e:
        results[name] = f"Failed with error: {e}"

# Display results
for algo, scores in results.items():
    print(f"\nAlgorithm: {algo}")
    if isinstance(scores, dict):
        for metric, score in scores.items():
            print(f"{metric}: {score}")
    else:
        print(scores)

# Evaluate clustering with different metrics
silhouette_avg = silhouette_score(feature_data_scaled_df, labels)
db_index = davies_bouldin_score(feature_data_scaled_df, labels)
ch_score = calinski_harabasz_score(feature_data_scaled_df, labels)

# # Perform K-Means clustering
# k = 3  # Number of clusters (you can experiment with different values)
# kmeans = KMeans(n_clusters=k, random_state=42)
# kmeans.fit(mfcc_scaled_with_pca.iloc[:, 1:])  # Fit the model to all features except the song number

# cluster_labels = kmeans.labels_
# all_features_df.insert(1, 'cluster_name', cluster_labels)

# print(f'Silhouette Score: {silhouette_avg:.4f}')
# print(f'Davies-Bouldin Index: {db_index:.4f}')
# print(f'Calinski-Harabasz Score: {ch_score:.4f}')

# Save the DataFrame to a CSV file
all_features_df.to_csv('extracted_features.csv', index=False)
print("Features saved to extracted_features.csv")

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data['Feature'] = feature_data.columns
vif_data['VIF'] = [variance_inflation_factor(feature_data.values, i) for i in range(feature_data.shape[1])]

# Filter features with VIF > 1
vif_above_1 = vif_data[vif_data['VIF'] > 2]

# Print features with VIF above 1
print("Features with VIF above 1:")
vif_data.to_csv('vif_data.csv', index=False)


# Plot VIF values
plt.figure(figsize=(10, 8))
sns.barplot(x='VIF', y='Feature', data=vif_data, palette='viridis')
plt.title('Variance Inflation Factor (VIF) for Features')
plt.xlabel('VIF')
plt.ylabel('Features')
plt.show()

# Plot PCA-reduced data, colored by cluster
#plt.scatter(pca_components[:, 0], pca_components[:, 1], c=cluster_labels, cmap='viridis', s=50)
#plt.title('K-Means Clustering on Generated Features (PCA-reduced 2D projection)')
#plt.xlabel('Principal Component 1')
#plt.ylabel('Principal Component 2')
#plt.colorbar(label='Cluster Label')
#plt.show()
