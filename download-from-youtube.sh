#!/bin/bash

# List of YouTube URLs
urls=(
    "https://www.youtube.com/watch?v=kEUXdnyXXO0"
    "https://www.youtube.com/watch?v=46GGxF_Bwhg"
    "https://www.youtube.com/watch?v=pAyKJAtDNCw"
    "https://www.youtube.com/watch?v=HtMF973tXIY"
    "https://www.youtube.com/watch?v=8xx-cFTF4XA"
    "https://www.youtube.com/watch?v=5JvML1N0S2E"
)

# Loop through the URLs and download each
for url in "${urls[@]}"; do
    # Get the song title without downloading, to use it in the message
    song_name=$(yt-dlp --get-title "$url" 2>/dev/null)

    # Download audio in WAV format, suppressing yt-dlp output
    yt-dlp -x --audio-format wav -o "%(title)s.%(ext)s" "$url" >/dev/null 2>&1

    # Print the download confirmation message
    echo "Downloaded ${song_name}"
done
