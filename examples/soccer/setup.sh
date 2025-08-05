#!/bin/bash

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if 'data' directory does not exist and then create it
if [[ ! -e $DIR/data ]]; then
    mkdir "$DIR/data"
    echo "Created 'data' directory."
else
    echo "'data' directory already exists."
fi

echo "Downloading pre-trained models..."

# Download the models
echo "Downloading football ball detection model..."
gdown -O "$DIR/data/football-ball-detection.pt" "https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V"

echo "Downloading football player detection model..."
gdown -O "$DIR/data/football-player-detection.pt" "https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q"

echo "Downloading football pitch detection model..."
gdown -O "$DIR/data/football-pitch-detection.pt" "https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"

echo "Downloading sample videos..."

# Download sample videos
echo "Downloading sample video 1..."
gdown -O "$DIR/data/0bfacc_0.mp4" "https://drive.google.com/uc?id=12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF"

echo "Downloading sample video 2..."
gdown -O "$DIR/data/2e57b9_0.mp4" "https://drive.google.com/uc?id=19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf"

echo "Downloading sample video 3..."
gdown -O "$DIR/data/08fd33_0.mp4" "https://drive.google.com/uc?id=1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-"

echo "Downloading sample video 4..."
gdown -O "$DIR/data/573e61_0.mp4" "https://drive.google.com/uc?id=1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU"

echo "Downloading sample video 5..."
gdown -O "$DIR/data/121364_0.mp4" "https://drive.google.com/uc?id=1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu"

echo "Setup completed! All models and sample videos have been downloaded."
echo "You can now run the main analysis script."