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

    # Get the song title without downloading, to use it in the message
    song_name=$(yt-dlp --get-title "$url" 2>/dev/null)

    # Download audio in WAV format, suppressing yt-dlp output
    yt-dlp -x --audio-format wav -o "%(title)s.%(ext)s" "$url" >/dev/null 2>&1

    # Print the download confirmation message
    echo "downloaded ${song_name}"
done
