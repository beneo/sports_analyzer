"""Base configuration classes using Pydantic for type safety."""

from typing import Optional, Union, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field, validator


class BaseConfig(BaseModel):
    """Base configuration class with common settings."""
    
    # Project settings
    project_name: str = Field(default="sports_analyzer", description="Project name")
    version: str = Field(default="0.1.0", description="Project version")
    
    # Logging settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="{time} | {level} | {name}:{function}:{line} - {message}",
        description="Log message format"
    )
    
    # Device settings
    device: str = Field(default="cpu", description="Computation device")
    
    @validator('device')
    def validate_device(cls, v):
        """Validate device setting."""
        allowed_devices = ["cpu", "cuda", "mps", "auto"]
        if v not in allowed_devices:
            raise ValueError(f"Device must be one of {allowed_devices}")
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of {allowed_levels}")
        return v.upper()
    
    class Config:
        """Pydantic config."""
        use_enum_values = True
        validate_assignment = True


class ModelConfig(BaseConfig):
    """Configuration for machine learning models."""
    
    # Model paths
    model_path: Optional[Path] = Field(None, description="Path to model file")
    checkpoint_path: Optional[Path] = Field(None, description="Path to checkpoint")
    
    # Model parameters
    batch_size: int = Field(default=32, ge=1, description="Batch size for inference")
    confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence threshold"
    )
    
    # SiglipVisionModel settings
    siglip_model_name: str = Field(
        default="google/siglip-base-patch16-224", 
        description="SiglipVisionModel identifier"
    )
    
    # UMAP settings
    umap_n_components: int = Field(default=3, ge=2, description="UMAP dimensions")
    umap_random_state: int = Field(default=42, description="UMAP random state")
    
    # KMeans settings
    kmeans_n_clusters: int = Field(default=2, ge=2, description="Number of clusters")
    kmeans_random_state: int = Field(default=42, description="KMeans random state")
    kmeans_n_init: int = Field(default=10, ge=1, description="KMeans initializations")


class TrainingConfig(BaseConfig):
    """Configuration for model training."""
    
    # Data settings
    data_path: Path = Field(..., description="Path to training data")
    output_path: Path = Field(default=Path("outputs"), description="Output directory")
    
    # Training parameters
    epochs: int = Field(default=100, ge=1, description="Number of training epochs")
    learning_rate: float = Field(default=0.001, gt=0.0, description="Learning rate")
    
    # Validation settings
    validation_split: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Validation data split"
    )
    
    # Checkpointing
    save_every: int = Field(default=10, ge=1, description="Save checkpoint every N epochs")
    early_stopping_patience: int = Field(
        default=20, ge=1, description="Early stopping patience"
    )
    
    @validator('data_path', 'output_path')
    def validate_paths(cls, v):
        """Ensure paths are Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v


class VideoProcessingConfig(BaseConfig):
    """Configuration for video processing."""
    
    # Input/Output
    input_video_path: Optional[Path] = Field(None, description="Input video path")
    output_video_path: Optional[Path] = Field(None, description="Output video path")
    
    # Processing settings
    fps: Optional[int] = Field(None, ge=1, description="Output FPS (None for original)")
    resolution: Optional[tuple] = Field(None, description="Output resolution (W, H)")
    quality: int = Field(default=95, ge=1, le=100, description="Output quality")
    
    # Frame processing
    start_frame: int = Field(default=0, ge=0, description="Start frame")
    end_frame: Optional[int] = Field(None, ge=0, description="End frame")
    frame_skip: int = Field(default=1, ge=1, description="Process every N frames")
    
    # Visualization settings
    show_annotations: bool = Field(default=True, description="Show annotations")
    annotation_thickness: int = Field(default=2, ge=1, description="Annotation thickness")
    
    @validator('input_video_path', 'output_video_path')
    def validate_video_paths(cls, v):
        """Ensure video paths are Path objects."""
        if v is not None and isinstance(v, str):
            return Path(v)
        return v
    
    @validator('resolution')
    def validate_resolution(cls, v):
        """Validate resolution format."""
        if v is not None:
            if not isinstance(v, (tuple, list)) or len(v) != 2:
                raise ValueError("Resolution must be a tuple/list of (width, height)")
            if not all(isinstance(x, int) and x > 0 for x in v):
                raise ValueError("Resolution values must be positive integers")
        return v


def load_config(config_path: Union[str, Path], config_class=BaseConfig) -> BaseConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        config_class: Configuration class to use
        
    Returns:
        Loaded configuration instance
    """
    from omegaconf import OmegaConf
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML config
    yaml_config = OmegaConf.load(config_path)
    
    # Convert to dict and create Pydantic model
    config_dict = OmegaConf.to_container(yaml_config, resolve=True)
    return config_class(**config_dict)


def save_config(config: BaseConfig, config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration instance to save
        config_path: Path to save configuration file
    """
    from omegaconf import OmegaConf
    
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict and save as YAML
    config_dict = config.dict()
    yaml_config = OmegaConf.create(config_dict)
    OmegaConf.save(yaml_config, config_path)