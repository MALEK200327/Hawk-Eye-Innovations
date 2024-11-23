#!/bin/bash

# Loop through each line of sports0_train.txt to download videos
while IFS= read -r url
do
    # Extract only the video URL (ignoring the label)
    video_url=$(echo $url | cut -d' ' -f1)
    yt-dlp "$video_url"
done < sports-1m-dataset/cross-validation/sports0_train.txt
