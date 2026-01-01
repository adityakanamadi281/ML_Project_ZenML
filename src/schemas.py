"""Data schemas for type safety and validation"""
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd


class ModelConfig(BaseModel):
    """Model configuration schema"""
    name: str
    version: str
    algorithm: str
    random_state: int = 42


class TrainingConfig(BaseModel):
    """Training configuration schema"""
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    cv_folds: int = 5


class DataConfig(BaseModel):
    """Data configuration schema"""
    source_path: str
    processed_path: str
    features_path: str
    target_column: str
    categorical_features: List[str] = []
    numerical_features: List[str] = []
    date_features: List[str] = []


class ModelMetrics(BaseModel):
    """Model evaluation metrics schema"""
    mse: float
    rmse: float
    mae: float
    r2_score: float
    model_name: str
    algorithm: str

