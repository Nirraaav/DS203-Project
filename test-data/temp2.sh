#!/bin/bash

# Loop through each .wav file with spaces in its name
for file in *.wav; do
    # Check if the filename contains spaces
    if [[ "$file" == *" "* ]]; then
        # Create the new filename by replacing spaces with hyphens
        new_file="${file// /-}"
        
        # Rename the file
        mv "$file" "$new_file"
    fi
done
