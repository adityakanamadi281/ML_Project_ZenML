"""Utility functions for the ML system"""
import yaml
from pathlib import Path
from typing import Dict, Any
from src.schemas import ModelConfig, TrainingConfig, DataConfig


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_model_config(config: Dict[str, Any]) -> ModelConfig:
    """Extract model configuration"""
    return ModelConfig(**config.get('model', {}))


def get_training_config(config: Dict[str, Any]) -> TrainingConfig:
    """Extract training configuration"""
    return TrainingConfig(**config.get('training', {}))


def get_data_config(config: Dict[str, Any]) -> DataConfig:
    """Extract data configuration"""
    data_config = config.get('data', {})
    features_config = config.get('features', {})
    return DataConfig(
        source_path=data_config.get('source_path', ''),
        processed_path=data_config.get('processed_path', ''),
        features_path=data_config.get('features_path', ''),
        target_column=features_config.get('target_column', 'price'),
        categorical_features=features_config.get('categorical_features', []),
        numerical_features=features_config.get('numerical_features', []),
        date_features=features_config.get('date_features', [])
    )


def ensure_directory(path: str) -> Path:
    """Ensure directory exists, create if not"""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

