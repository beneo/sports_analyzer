# Sports Analyzer

A comprehensive sports analytics toolkit built with computer vision and deep learning technologies. This project focuses on pushing the limits of object detection, image segmentation, keypoint detection, and foundational models in sports scenarios.

## 🚀 Features

- **Player Detection & Tracking**: Advanced multi-object tracking for sports players
- **Ball Tracking**: High-precision ball detection and trajectory analysis  
- **Team Classification**: Automatic team identification using visual embeddings
- **Field Analysis**: Soccer pitch keypoint detection and coordinate transformation
- **Real-time Visualization**: Rich annotations and analytics overlays

## 🏆 Challenges Addressed

- **Ball tracking**: Tracking small, fast-moving objects in high-resolution videos
- **Jersey number reading**: OCR for player identification despite blur and occlusion
- **Player tracking**: Consistent player ID maintenance through occlusions
- **Player re-identification**: Re-identifying players who leave and re-enter the frame
- **Camera calibration**: Accurate view transformation for advanced statistics

## 💻 Installation

### From Source
Install in a Python>=3.8 environment:

```bash
# Clone and install
cd sports_analyzer
pip install -e .
```

### Example Dependencies
For running soccer examples:

```bash
cd examples/soccer
pip install -r requirements.txt
./setup.sh  # Download pre-trained models
```

## 🎮 Quick Start

### Basic Usage

```python
from sports_analyzer.common import BallTracker, TeamClassifier
from sports_analyzer.configs.soccer import SoccerPitchConfiguration
from sports_analyzer.annotators.soccer import draw_pitch

# Initialize components
ball_tracker = BallTracker(buffer_size=10)
team_classifier = TeamClassifier(device='cpu')
pitch_config = SoccerPitchConfiguration()

# Create pitch visualization
pitch_image = draw_pitch(pitch_config)
```

### Soccer Video Analysis

```bash
cd examples/soccer

# Player tracking mode
python main.py \
  --source_video_path input.mp4 \
  --target_video_path output.mp4 \
  --mode PLAYER_TRACKING

# Ball tracking mode  
python main.py \
  --source_video_path input.mp4 \
  --target_video_path output.mp4 \
  --mode BALL_TRACKING

# Radar view analysis
python main.py \
  --source_video_path input.mp4 \
  --target_video_path output.mp4 \
  --mode RADAR
```

## 🏗️ Architecture

### Core Modules

- **`sports_analyzer.common`**: Core algorithms and utilities
  - `ball.py`: Ball tracking and annotation
  - `team.py`: Team classification using SiglipVisionModel
  - `view.py`: Homography-based coordinate transformations

- **`sports_analyzer.configs`**: Configuration classes
  - `soccer.py`: Soccer pitch dimensions and layouts

- **`sports_analyzer.annotators`**: Visualization utilities  
  - `soccer.py`: Pitch drawing and overlay functions

### Technology Stack

- **Computer Vision**: OpenCV, supervision framework
- **Deep Learning**: transformers (SiglipVisionModel), ultralytics (YOLO)
- **Machine Learning**: UMAP dimensionality reduction, KMeans clustering
- **Core Libraries**: numpy, scikit-learn, tqdm

## 🔧 Development

### Project Structure
```
sports_analyzer/
├── sports_analyzer/           # Core package
│   ├── common/               # Core algorithms
│   ├── configs/              # Configurations  
│   └── annotators/           # Visualizations
├── examples/                 # Usage examples
│   └── soccer/              # Soccer analysis
├── setup.py                 # Package setup
└── README.md               # Documentation
```

### Testing
```bash
pytest
```

## 📊 Datasets

The project works with various sports datasets:
- Soccer player detection
- Soccer ball detection  
- Soccer pitch keypoint detection
- Basketball court keypoint detection
- Basketball jersey numbers OCR

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines for more details.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.