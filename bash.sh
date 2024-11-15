#!/bin/bash

# Directory where files are located
dir="jgm-mfcc"

# Counter for renaming files
counter=1

# Loop through all files in the directory
for file in "$dir"/*; do
    # Get the extension of the file
    ext="${file##*.}"

    # Rename the file
    mv "$file" "$dir/Jana-Gana-Mana-$counter.$ext"

    # Increment the counter
    ((counter++))
done

echo "Renaming completed!"
