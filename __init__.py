"""
Sports Analyzer - A comprehensive sports analytics toolkit.

This package provides computer vision and deep learning tools for sports analysis,
with a focus on soccer/football analytics including player tracking, team classification,
ball tracking, and tactical analysis.
"""

__version__ = "0.1.0"
__author__ = "Sports Analytics Team"
__email__ = "team@sports-analyzer.com"

from .common import BallTracker, BallAnnotator, TeamClassifier, ViewTransformer
from .configs import SoccerPitchConfiguration, BaseConfig, ModelConfig
from .annotators import draw_pitch, draw_points_on_pitch, draw_paths_on_pitch

__all__ = [
    # Core components
    "BallTracker",
    "BallAnnotator", 
    "TeamClassifier",
    "ViewTransformer",
    
    # Configuration
    "SoccerPitchConfiguration",
    "BaseConfig",
    "ModelConfig",
    
    # Visualization
    "draw_pitch",
    "draw_points_on_pitch",
    "draw_paths_on_pitch",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__"
]