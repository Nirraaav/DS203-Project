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
        print(f"Error creating MFCC coefficients for {file_name}: {str(e)}")

# Function to process all audio files in the 'test-data' folder
def process_all_audio_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop over all files in the input folder
    for file_name in os.listdir(input_folder):
        # Check if the file is a .wav file
        if file_name.endswith('.wav'):
            file_path = os.path.join(input_folder, file_name)
            
            # Create MFCC coefficients
            coeff_df = create_MFCC_coefficients(file_path)
            
            if coeff_df is not None:
                # Create output file path
                output_file_path = os.path.join(output_folder, file_name.replace('.wav', '-MFCC.csv'))
                
                # Save MFCC to CSV
                coeff_df.to_csv(output_file_path, header=False, index=False)
                print(f"Processed {file_name} and saved to {output_file_path}")
            else:
                print(f"Skipping {file_name} due to error in processing.")
        else:
            print(f"Skipping non-audio file {file_name}")

# Folder paths
input_folder = 'test-data'
output_folder = 'test-mfcc'

# Process all files in the 'test-data' folder
process_all_audio_files(input_folder, output_folder)
