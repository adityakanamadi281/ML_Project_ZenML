"""Data preprocessing step"""
import pandas as pd
from zenml import step
from typing import Tuple
from src.utils import get_data_config, load_config


@step
def preprocess_data(
    data: pd.DataFrame,
    config_path: str = "config.yaml"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses the raw data: handles missing values, outliers, etc.
    
    Args:
        data: Raw DataFrame
        config_path: Path to configuration file
        
    Returns:
        Tuple of (features DataFrame, target Series as DataFrame)
    """
    config = load_config(config_path)
    data_config = get_data_config(config)
    
    df = data.copy()
    
    # Handle missing values
    print("🔧 Handling missing values...")
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
        # Fill numerical columns with median
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            if col != data_config.target_column:
                df[col].fillna(df[col].median(), inplace=True)
        # Fill categorical columns with mode
        for col in df.select_dtypes(include=['object']).columns:
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown', inplace=True)
    
    # Separate features and target
    if data_config.target_column not in df.columns:
        raise ValueError(f"Target column '{data_config.target_column}' not found in data")
    
    y = df[[data_config.target_column]]
    X = df.drop(columns=[data_config.target_column])
    
    print(f"✅ Data preprocessed. Features shape: {X.shape}, Target shape: {y.shape}")
    
    return X, y

