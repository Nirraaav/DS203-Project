import os
import numpy as np
import pandas as pd
import librosa

# Function to create MFCC coefficients given an audio file
def create_MFCC_coefficients(file_name):
    sr_value = 44100
    n_mfcc_count = 20
    
    try:
        # Load the audio file using librosa
        y, sr = librosa.load(file_name, sr=sr_value)
              
        # Compute MFCC coefficients for the segment
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc_count)
        
        # Create and return MFCC dataframe
        coeff_df = pd.DataFrame(mfccs)
        
        return coeff_df

    except Exception as e:
        print(f"Error creating MFCC coefficients: {file_name}: {str(e)}")

# Folder containing the audio files
audio_folder = "jgm"

# Output folder to save the MFCC CSV files
output_folder = "jgm-mfcc"

# Create output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the audio folder
for file_name in os.listdir(audio_folder):
    if file_name.endswith(".wav"):  # Assuming the files are in WAV format
        # Full path to the audio file
        audio_path = os.path.join(audio_folder, file_name)
        
        # Generate MFCC coefficients for the audio file
        mfcc_df = create_MFCC_coefficients(audio_path)
        
        if mfcc_df is not None:
            # Construct the output CSV file path
            output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.csv")
            
            # Save the MFCC coefficients to a CSV file
            mfcc_df.to_csv(output_file, index=False)
            print(f"Saved MFCC coefficients for {file_name} to {output_file}")
