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

def compute_zcr(audio_segment):
    zcr = librosa.feature.zero_crossing_rate(audio_segment)
    return np.mean(zcr)

def compute_rmse(audio_segment):
    rmse = librosa.feature.rms(y=audio_segment)
    return np.mean(rmse)

def compute_tempo(audio_segment, sr=44100):
    onset_env = librosa.onset.onset_strength(y=audio_segment, sr=sr)
    detected_tempo = compute_tempo_function(onset_envelope=onset_env, sr=sr)
    return detected_tempo[0]

def compute_spectral_centroid(audio_segment, sr=44100):
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)
    return np.mean(spectral_centroid)

def compute_spectral_bandwidth(audio_segment, sr=44100):
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_segment, sr=sr)
    return np.mean(spectral_bandwidth)

def compute_spectral_rolloff(audio_segment, sr=44100):
    rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=sr, roll_percent=0.85)
    return np.mean(rolloff)

def compute_chroma(audio_segment, sr=44100):
    chroma = librosa.feature.chroma_stft(y=audio_segment, sr=sr)
    return np.mean(chroma)

def compute_mfcc_mean(mfcc_data):
    return np.mean(mfcc_data, axis=1)

def compute_mfcc_std(mfcc_data):
    return np.std(mfcc_data, axis=1)

def compute_mfcc_delta(mfcc_data):
    mfcc_delta = librosa.feature.delta(mfcc_data)
    return np.mean(mfcc_delta)

def compute_mfcc_delta2(mfcc_data):
    mfcc_delta2 = librosa.feature.delta(mfcc_data, order=2)
    return np.mean(mfcc_delta2)

def compute_contrast(audio_segment, sr=44100):
    contrast = librosa.feature.spectral_contrast(y=audio_segment, sr=sr)
    return np.mean(contrast)

def compute_tonnetz(audio_segment, sr=44100):
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_segment), sr=sr)
    return np.mean(tonnetz)

# Initialize lists to store features and file names
mfcc_all_songs = []
file_names = []
generated_features = []

for i in range(1, 117):
    file_name = f'data-v2/{i:02d}-MFCC.csv'
    mfcc_data = pd.read_csv(file_name, header=None).values

    # Compute aggregated MFCC features
    aggregated_features = aggregate_mfcc_selective(mfcc_data)

    # Uncomment the following feature computations as needed
    # zcr = compute_zcr(audio_segment)
    # rmse = compute_rmse(audio_segment)
    # tempo = compute_tempo(audio_segment, sr)
    # spectral_centroid = compute_spectral_centroid(audio_segment, sr)
    # spectral_bandwidth = compute_spectral_bandwidth(audio_segment, sr)
    # spectral_rolloff = compute_spectral_rolloff(audio_segment, sr)
    # chroma = compute_chroma(audio_segment, sr)
    # mfcc_mean = compute_mfcc_mean(mfcc_data)
    # mfcc_std = compute_mfcc_std(mfcc_data)
    # mfcc_delta = compute_mfcc_delta(mfcc_data)
    # mfcc_delta2 = compute_mfcc_delta2(mfcc_data)
    skewness = skew(mfcc_data, axis=1)
    kurt = kurtosis(mfcc_data, axis=1)
    range_max_min = np.ptp(mfcc_data, axis=1)
    # contrast = compute_contrast(audio_segment, sr)
    # tonnetz = compute_tonnetz(audio_segment, sr)

    delta_mfcc = np.diff(mfcc_data, axis=1)
    delta_delta_mfcc = np.diff(delta_mfcc, axis=1)

    delta_mean = np.mean(delta_mfcc, axis=1)
    delta_std = np.std(delta_mfcc, axis=1)
    delta_max = np.max(delta_mfcc, axis=1)
    delta_min = np.min(delta_mfcc, axis=1)
    
    # mfcc_entropy = -np.sum((mfcc_data + 1e-6) * np.log(mfcc_data + 1e-6), axis=1) / mfcc_data.shape[1]

    PCA_mfcc = PCA(n_components=5)
    PCA_mfcc.fit(mfcc_data.T)
    PCA_mfcc_features = PCA_mfcc.components_
    PCA_mfcc_features = PCA_mfcc_features.flatten()

    # Compile all features into a single vector
    features = np.concatenate([
        aggregated_features.flatten(),
        # Uncomment the additional features you wish to include
        # zcr,
        # rmse,
        # tempo,
        # spectral_centroid,
        # spectral_bandwidth,
        # spectral_rolloff,
        # chroma,
        # mfcc_mean.mean(),
        # mfcc_std.mean(),
        # mfcc_delta.reshape(-1),
        # mfcc_delta2.reshape(-1),
        # contrast,
        # tonnetz,
        # skewness.flatten(),
        # kurt.flatten(),
        range_max_min.flatten(),
        delta_mean.flatten(),
        delta_std.flatten(),
        delta_max.flatten(),
        delta_min.flatten(),
        # mfcc_entropy.flatten(),
        PCA_mfcc_features.flatten(),
    ])

    # print(f'Processed {file_name}')
    
    generated_features.append(features)  # Append the features for this song
    file_names.append(file_name)          # Store the file name

generated_features = np.vstack(generated_features)
# print(generated_features.shape)  # Should print (number_of_songs, number_of_features)
total_features = generated_features.shape[1]
feature_columns = [f'feature_{i}' for i in range(total_features)]

features_df = pd.DataFrame(generated_features, columns=feature_columns)
features_df.insert(0, 'File', file_names)

# Save generated features to 'features_generated.csv'
features_df.to_csv(f'features_generated_{INDEX}.csv', index=False)

# Standardize the features
scaler = StandardScaler()
mfcc_scaled = scaler.fit_transform(generated_features)

# Perform K-Means clustering
k = 6  # Number of clusters (you can experiment with different values)
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(mfcc_scaled)

# Get cluster labels for each song
cluster_labels = kmeans.labels_

# Evaluate clustering with different metrics
silhouette_avg = silhouette_score(mfcc_scaled, cluster_labels)
db_index = davies_bouldin_score(mfcc_scaled, cluster_labels)
ch_score = calinski_harabasz_score(mfcc_scaled, cluster_labels)

print(f'Silhouette Score: {silhouette_avg:.4f}')
print(f'Davies-Bouldin Index: {db_index:.4f}')
print(f'Calinski-Harabasz Score: {ch_score:.4f}')

# Output the cluster label for each file
print("\nCluster Labels for Each File:")
for file_name, label in zip(file_names, cluster_labels):
    print(f'{file_name}: Cluster {label}')

# Visualize clusters using PCA for 2D projection
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(mfcc_scaled)

# Plot PCA-reduced data, colored by cluster
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis', s=50)
plt.title('K-Means Clustering on Generated Features (PCA-reduced 2D projection)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.savefig('cluster_plot.png')
plt.show()

# Save file names and cluster labels to 'file_cluster_labels.csv'
output_df = pd.DataFrame({
    'File': file_names,
    'Cluster': cluster_labels
})
output_df.to_csv(f'labels/file_cluster_labels_{INDEX}.csv', index=False)