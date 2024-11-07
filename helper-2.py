import os
import numpy as np
import pandas as pd

# Helper function to append three rows of means
def append_means(mfcc_data):
    # Compute means for specified row ranges
    mean_1_3 = np.mean(mfcc_data[0:3, :], axis=0)    # Mean of rows 1 to 3
    mean_4_13 = np.mean(mfcc_data[3:13, :], axis=0)  # Mean of rows 4 to 13
    mean_14_20 = np.mean(mfcc_data[13:20, :], axis=0) # Mean of rows 14 to 20

    # Stack these mean rows as new rows
    mean_rows = np.vstack([mean_1_3, mean_4_13, mean_14_20])
    
    # Append the new rows to the original mfcc data
    mfcc_data_extended = np.vstack([mfcc_data, mean_rows])
    
    return mfcc_data_extended

# Loop through all CSV files in the test directory (assuming you have a test directory)
test_directory = 'data-v2-copy'
test_files = [file for file in os.listdir(test_directory) if file.endswith("01-MFCC.csv")]
test_files.sort()  # Sort the files alphabetically

for test_file in test_files:
    # test_file_path = os.path.join(test_directory, test_file)
    # mfcc_data_test = pd.read_csv(test_file_path, header=None).values
    
    # # Append the mean rows
    # mfcc_data_test_extended = append_means(mfcc_data_test)
    
    # # Save the modified test file with the appended rows
    # extended_file_path = test_file_path.replace('.csv', '_extended.csv')
    # pd.DataFrame(mfcc_data_test_extended).to_csv(extended_file_path, header=False, index=False)

    # print(f'Processed {test_file} and saved extended file as {extended_file_path}')   
    
    # # delete files ending with -MFCC.csv
    # os.remove(os.path.join(test_directory, test_file))

    # # remove _extended from the filenames
    # new_filename = os.path.join(test_directory, test_file.replace('_extended', ''))
    # os.rename(os.path.join(test_directory, test_file), new_filename)
    # print(f'Renamed {test_file} to {new_filename}')

    # delete the first 20 rows for each file and keep only the last 3 rows
    test_file_path = os.path.join(test_directory, test_file)
    mfcc_data_test = pd.read_csv(test_file_path, header=None).values
    mfcc_data_test = mfcc_data_test[-3:, :]
    pd.DataFrame(mfcc_data_test).to_csv(test_file_path, header=False, index=False)
    print(f'Processed {test_file} and saved file with only the last 3 rows')

