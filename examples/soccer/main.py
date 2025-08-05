#!/usr/bin/env python3
"""
Soccer Analysis Main Script

This script provides a comprehensive soccer video analysis pipeline with multiple modes:
- PLAYER_TRACKING: Track players with bounding boxes and IDs
- BALL_TRACKING: Track ball with trajectory visualization  
- TEAM_CLASSIFICATION: Classify players into teams using visual features
- RADAR: Generate top-down radar view analysis
- HEATMAP: Create player position heatmaps

Usage:
    python main.py --source_video_path input.mp4 --target_video_path output.mp4 --mode PLAYER_TRACKING
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from loguru import logger
from rich.console import Console
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
from tqdm import tqdm
from ultralytics import YOLO

# Add the sports_analyzer package to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sports_analyzer.common import BallTracker, BallAnnotator, TeamClassifier, ViewTransformer
from sports_analyzer.configs import SoccerPitchConfiguration, VideoProcessingConfig
from sports_analyzer.annotators import (
    draw_pitch, draw_points_on_pitch, draw_paths_on_pitch, 
    draw_pitch_voronoi_diagram, draw_heatmap_on_pitch
)

console = Console()


class SoccerAnalyzer:
    """Main soccer analysis pipeline."""
    
    def __init__(
        self, 
        player_model_path: str,
        ball_model_path: str, 
        pitch_model_path: str,
        device: str = "cpu"
    ):
        """
        Initialize the soccer analyzer with all required models.
        
        Args:
            player_model_path: Path to player detection model
            ball_model_path: Path to ball detection model  
            pitch_model_path: Path to pitch keypoint detection model
            device: Device for inference
        """
        self.device = device
        
        # Load YOLO models
        logger.info("Loading detection models...")
        self.player_model = YOLO(player_model_path)
        self.ball_model = YOLO(ball_model_path)
        self.pitch_model = YOLO(pitch_model_path)
        
        # Initialize trackers and annotators
        self.ball_tracker = BallTracker(buffer_size=10)
        self.ball_annotator = BallAnnotator(radius=15, buffer_size=5)
        
        # Initialize supervision trackers
        self.byte_tracker = sv.ByteTrack()
        
        # Initialize annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.ellipse_annotator = sv.EllipseAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
        
        # Configuration
        self.pitch_config = SoccerPitchConfiguration()
        
        # Team classifier (initialized on demand)
        self.team_classifier: Optional[TeamClassifier] = None
        
        # View transformer (initialized when pitch is detected)
        self.view_transformer: Optional[ViewTransformer] = None
        
        logger.info("Soccer analyzer initialized successfully!")

    def detect_players(self, frame: np.ndarray) -> sv.Detections:
        """Detect players in the frame."""
        results = self.player_model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        return detections

    def detect_ball(self, frame: np.ndarray) -> sv.Detections:
        """Detect ball in the frame."""
        results = self.ball_model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        return detections

    def detect_pitch_keypoints(self, frame: np.ndarray) -> sv.Detections:
        """Detect pitch keypoints."""
        results = self.pitch_model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        return detections

    def setup_team_classifier(self, frames: List[np.ndarray], max_samples: int = 100) -> None:
        """
        Setup team classifier by collecting player crops from sample frames.
        
        Args:
            frames: Sample frames for training
            max_samples: Maximum number of player crops to collect
        """
        logger.info("Setting up team classifier...")
        
        player_crops = []
        
        for frame in frames[:10]:  # Use first 10 frames
            detections = self.detect_players(frame)
            
            if len(detections) > 0:
                # Get player crops
                for detection in detections:
                    xyxy = detection.xyxy[0].astype(int)
                    crop = frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                    
                    if crop.size > 0:
                        player_crops.append(crop)
                        
                    if len(player_crops) >= max_samples:
                        break
                        
            if len(player_crops) >= max_samples:
                break
        
        if len(player_crops) < 10:
            logger.warning("Insufficient player crops for team classification")
            return
            
        # Initialize and train team classifier
        self.team_classifier = TeamClassifier(device=self.device)
        self.team_classifier.fit(player_crops)
        
        logger.info(f"Team classifier trained on {len(player_crops)} player crops")

    def process_frame_player_tracking(self, frame: np.ndarray) -> np.ndarray:
        """Process frame for player tracking mode."""
        # Detect and track players
        detections = self.detect_players(frame)
        tracked_detections = self.byte_tracker.update_with_detections(detections)
        
        # Create labels with tracker IDs
        labels = [f"Player {tracker_id}" for tracker_id in tracked_detections.tracker_id]
        
        # Annotate frame
        annotated_frame = self.ellipse_annotator.annotate(frame.copy(), tracked_detections)
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, tracked_detections, labels=labels
        )
        
        return annotated_frame

    def process_frame_ball_tracking(self, frame: np.ndarray) -> np.ndarray:
        """Process frame for ball tracking mode."""
        # Detect and track ball
        detections = self.detect_ball(frame)
        tracked_ball = self.ball_tracker.update(detections)
        
        # Annotate frame with ball trajectory
        annotated_frame = self.ball_annotator.annotate(frame.copy(), tracked_ball)
        
        return annotated_frame

    def process_frame_team_classification(self, frame: np.ndarray) -> np.ndarray:
        """Process frame for team classification mode."""
        if self.team_classifier is None:
            return frame
            
        # Detect and track players
        detections = self.detect_players(frame)
        tracked_detections = self.byte_tracker.update_with_detections(detections)
        
        if len(tracked_detections) == 0:
            return frame
        
        # Get player crops for classification
        player_crops = []
        for detection in tracked_detections:
            xyxy = detection.xyxy[0].astype(int)
            crop = frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
            if crop.size > 0:
                player_crops.append(crop)
        
        if len(player_crops) == 0:
            return frame
            
        # Classify teams
        team_predictions = self.team_classifier.predict(player_crops)
        
        # Create color-coded annotations
        colors = [sv.Color.RED if pred == 0 else sv.Color.BLUE for pred in team_predictions]
        labels = [f"Team {pred + 1}" for pred in team_predictions]
        
        # Annotate frame
        annotated_frame = frame.copy()
        for i, (detection, color, label) in enumerate(zip(tracked_detections, colors, labels)):
            # Draw colored ellipse
            sv.EllipseAnnotator(color=color, thickness=3).annotate(
                annotated_frame, sv.Detections(xyxy=detection.xyxy[i:i+1])
            )
            # Add label
            sv.LabelAnnotator(text_color=color).annotate(
                annotated_frame, sv.Detections(xyxy=detection.xyxy[i:i+1]), labels=[label]
            )
        
        return annotated_frame

    def process_frame_radar(self, frame: np.ndarray) -> np.ndarray:
        """Process frame for radar view mode."""
        # Detect players
        player_detections = self.detect_players(frame)
        
        if len(player_detections) == 0:
            # Return radar view with no players
            radar_frame = draw_pitch(self.pitch_config, scale=0.15, padding=100)
            return radar_frame
        
        # Get player positions (bottom center of bounding boxes)
        player_positions = player_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        
        # If we have a view transformer, convert to field coordinates
        if self.view_transformer is not None:
            try:
                field_positions = self.view_transformer.transform_points(player_positions)
            except:
                field_positions = player_positions  # Fallback to original positions
        else:
            # Use mock field coordinates for demonstration
            h, w = frame.shape[:2]
            field_positions = []
            for pos in player_positions:
                # Simple mapping from image to field coordinates
                field_x = (pos[0] / w) * self.pitch_config.length
                field_y = (pos[1] / h) * self.pitch_config.width
                field_positions.append([field_x, field_y])
            field_positions = np.array(field_positions)
        
        # Create radar view
        radar_frame = draw_pitch(self.pitch_config, scale=0.15, padding=100)
        
        # Draw player positions on radar
        if len(field_positions) > 0:
            radar_frame = draw_points_on_pitch(
                self.pitch_config,
                field_positions,
                face_color=sv.Color.RED,
                edge_color=sv.Color.WHITE,
                radius=8,
                scale=0.15,
                padding=100,
                pitch=radar_frame
            )
        
        return radar_frame

    def process_frame_heatmap(self, frame: np.ndarray, position_history: List[np.ndarray]) -> np.ndarray:
        """Process frame for heatmap mode."""
        # Detect current player positions
        player_detections = self.detect_players(frame)
        
        if len(player_detections) > 0:
            player_positions = player_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            position_history.append(player_positions)
        
        # Limit history size
        if len(position_history) > 100:
            position_history.pop(0)
        
        if not position_history:
            return draw_pitch(self.pitch_config, scale=0.15, padding=100)
        
        # Combine all positions
        all_positions = np.vstack(position_history)
        
        # Convert to field coordinates (simplified mapping)
        h, w = frame.shape[:2]
        field_positions = []
        for pos in all_positions:
            field_x = (pos[0] / w) * self.pitch_config.length
            field_y = (pos[1] / h) * self.pitch_config.width
            field_positions.append([field_x, field_y])
        field_positions = np.array(field_positions)
        
        # Create heatmap
        heatmap_frame = draw_heatmap_on_pitch(
            self.pitch_config,
            field_positions,
            scale=0.15,
            padding=100,
            alpha=0.6
        )
        
        return heatmap_frame

    def analyze_video(
        self, 
        source_path: str, 
        target_path: str, 
        mode: str = "PLAYER_TRACKING"
    ) -> None:
        """
        Analyze video with specified mode.
        
        Args:
            source_path: Input video path
            target_path: Output video path
            mode: Analysis mode
        """
        logger.info(f"Starting video analysis in {mode} mode")
        logger.info(f"Input: {source_path}")
        logger.info(f"Output: {target_path}")
        
        # Open video
        video_info = sv.VideoInfo.from_video_path(source_path)
        
        with sv.VideoSink(target_path, video_info) as sink:
            # Setup team classifier if needed
            if mode == "TEAM_CLASSIFICATION":
                # Read sample frames for training
                frames = []
                frame_generator = sv.get_video_frames_generator(source_path)
                for i, frame in enumerate(frame_generator):
                    frames.append(frame)
                    if i >= 20:  # Get first 20 frames
                        break
                
                self.setup_team_classifier(frames)
            
            # Initialize mode-specific variables
            position_history = []
            
            # Process video
            frame_generator = sv.get_video_frames_generator(source_path)
            total_frames = video_info.total_frames
            
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                
                task = progress.add_task(f"Processing {mode}...", total=total_frames)
                
                for frame in frame_generator:
                    # Process frame based on mode
                    if mode == "PLAYER_TRACKING":
                        processed_frame = self.process_frame_player_tracking(frame)
                    elif mode == "BALL_TRACKING":
                        processed_frame = self.process_frame_ball_tracking(frame)
                    elif mode == "TEAM_CLASSIFICATION":
                        processed_frame = self.process_frame_team_classification(frame)
                    elif mode == "RADAR":
                        processed_frame = self.process_frame_radar(frame)
                    elif mode == "HEATMAP":
                        processed_frame = self.process_frame_heatmap(frame, position_history)
                    else:
                        processed_frame = frame
                    
                    # Write frame
                    sink.write_frame(processed_frame)
                    progress.advance(task)
        
        logger.info(f"Video analysis completed! Output saved to: {target_path}")


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Soccer Video Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available modes:
  PLAYER_TRACKING     - Track players with bounding boxes and IDs
  BALL_TRACKING       - Track ball with trajectory visualization
  TEAM_CLASSIFICATION - Classify players into teams using visual features
  RADAR              - Generate top-down radar view analysis  
  HEATMAP            - Create player position heatmaps

Examples:
  python main.py --source_video_path data/sample.mp4 --target_video_path output.mp4 --mode PLAYER_TRACKING
  python main.py --source_video_path data/sample.mp4 --target_video_path output.mp4 --mode TEAM_CLASSIFICATION --device cuda
        """
    )
    
    parser.add_argument(
        "--source_video_path",
        type=str,
        required=True,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--target_video_path", 
        type=str,
        required=True,  
        help="Path to output video file"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="PLAYER_TRACKING",
        choices=["PLAYER_TRACKING", "BALL_TRACKING", "TEAM_CLASSIFICATION", "RADAR", "HEATMAP"],
        help="Analysis mode"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference (cpu, cuda, mps)"
    )
    
    parser.add_argument(
        "--player_model_path",
        type=str,
        default="data/football-player-detection.pt",
        help="Path to player detection model"
    )
    
    parser.add_argument(
        "--ball_model_path",
        type=str, 
        default="data/football-ball-detection.pt",
        help="Path to ball detection model"
    )
    
    parser.add_argument(
        "--pitch_model_path",
        type=str,
        default="data/football-pitch-detection.pt", 
        help="Path to pitch keypoint detection model"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.source_video_path).exists():
        console.print(f"[red]Error: Source video not found: {args.source_video_path}[/red]")
        return 1
        
    for model_path in [args.player_model_path, args.ball_model_path, args.pitch_model_path]:
        if not Path(model_path).exists():
            console.print(f"[red]Error: Model not found: {model_path}[/red]")
            console.print("[yellow]Please run './setup.sh' to download required models[/yellow]")
            return 1
    
    # Create output directory
    Path(args.target_video_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize analyzer
        analyzer = SoccerAnalyzer(
            player_model_path=args.player_model_path,
            ball_model_path=args.ball_model_path,
            pitch_model_path=args.pitch_model_path,
            device=args.device
        )
        
        # Analyze video
        analyzer.analyze_video(
            source_path=args.source_video_path,
            target_path=args.target_video_path,
            mode=args.mode
        )
        
        console.print(f"[green]✅ Analysis completed successfully![/green]")
        console.print(f"[blue]Output video: {args.target_video_path}[/blue]")
        
        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"[red]Error during analysis: {e}[/red]")
        logger.exception("Analysis failed")
        return 1


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    sys.exit(main())