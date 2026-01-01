"""Prediction/inference step"""
import pandas as pd
from zenml import step
from typing import List
import joblib
from pathlib import Path
from src.utils import get_model_config, load_config


@step
def predict(
    X: pd.DataFrame,
    config_path: str = "config.yaml"
) -> pd.DataFrame:
    """
    Makes predictions using the trained model.
    
    Args:
        X: Features DataFrame for prediction
        config_path: Path to configuration file
        
    Returns:
        DataFrame with predictions
    """
    config = load_config(config_path)
    model_config = get_model_config(config)
    
    # Load model
    model_path = f"models/{model_config.name}_{model_config.version}.joblib"
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    model = joblib.load(model_path)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'predicted_price': predictions
    })
    
    print(f"✅ Generated {len(predictions)} predictions")
    
    return results

