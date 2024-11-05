#!/bin/bash

# Infinite loop to keep asking for URLs
while true; do
    # Prompt user for a URL
    read -p "Enter a YouTube URL (or type 'exit' to quit): " url

    # Exit the loop if the user types "exit"
    if [[ "$url" == "exit" ]]; then
        echo "Exiting..."
        break
    fi

    # Download audio in WAV format with yt-dlp
    yt-dlp -x --audio-format wav -o "%(title)s.%(ext)s" "$url"
done
