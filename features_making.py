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

INDEX = 1  # Start index for MFCC features you want to include

def extract_features(mfcc_data):
    # Select the MFCC data starting from the specified INDEX
    mfcc_selected = mfcc_data[INDEX:, :]
    
    # Calculate features for each MFCC coefficient
    features = {
        'mean': np.mean(mfcc_selected, axis=1),
        'std': np.std(mfcc_selected, axis=1),
        'max': np.max(mfcc_selected, axis=1),
        'min': np.min(mfcc_selected, axis=1),
        'delta': np.mean(librosa.feature.delta(mfcc_selected), axis=1),
        'delta2': np.mean(librosa.feature.delta(mfcc_selected, order=2), axis=1),
        'delta_max': np.max(librosa.feature.delta(mfcc_selected), axis=1),
        'delta2_max': np.max(librosa.feature.delta(mfcc_selected, order=2), axis=1),
        'delta_min': np.min(librosa.feature.delta(mfcc_selected), axis=1),
        'delta2_min': np.min(librosa.feature.delta(mfcc_selected, order=2), axis=1),
        'skew': skew(mfcc_selected, axis=1),
        'kurtosis': kurtosis(mfcc_selected, axis=1),
        'range': np.ptp(mfcc_selected, axis=1),
        'total_energy': np.sum(mfcc_selected ** 2, axis=1),
        'energy_entropy': -np.sum(mfcc_selected ** 2 * np.log(mfcc_selected ** 2 + 1e-10), axis=1)
    }
    return features

# Define DataFrame columns
columns = []
for stat in ['mean', 'std', 'max', 'min', 'delta', 'delta2', 'delta_max', 'delta2_max', 'delta_min', 'delta_max','skew', 'kurtosis', 'range']:
    columns.extend([f'{stat}_mfcc_{i}' for i in range(INDEX, 20)])  # assuming MFCCs from INDEX to 19

# Initialize an empty DataFrame to store all features
all_features_df = pd.DataFrame(columns=['song_number'] + columns)

for i in range(1, 117):
    file_name = f'data-v2/{i:02d}-MFCC.csv'
    mfcc_data = pd.read_csv(file_name, header=None).values

    # Extract features
    features = extract_features(mfcc_data)
    
    # Flatten the feature dictionary into a list
    row = [f'song_{i}'] + [features[stat][j] for stat in features for j in range(len(features[stat]))]
    
    # Append the row to the DataFrame
    all_features_df.loc[i - 1] = row

# Generate the correlation matrix
heatmap_data = all_features_df.drop(columns=['song_number'])
correlation_matrix = heatmap_data.corr()

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




# Perform K-Means clustering
k = 6  # Number of clusters (you can experiment with different values)
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(all_features_df.iloc[:, 1:])  # Fit the model to all features except the song number

cluster_labels = kmeans.labels_
all_features_df.insert(1, 'cluster_name', cluster_labels)

# Save the DataFrame to a CSV file
all_features_df.to_csv('extracted_features.csv', index=False)
print("Features saved to extracted_features.csv")



