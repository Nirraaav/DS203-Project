#!/bin/bash

# Loop through each file that matches the pattern "Asha Bhosale *.wav"
for file in Asha\ Bhosale\ *.wav; do
    # Create the new filename by replacing "Asha Bhosale" with "Asha Bhosle"
    new_file="${file/Bhosale/Bhosle}"
    
    # Rename the file
    mv "$file" "$new_file"
done
