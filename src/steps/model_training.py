"""Model training step"""
import pandas as pd
from zenml import step
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
from src.utils import get_model_config, get_training_config, load_config


@step
def train_model(
    X: pd.DataFrame,
    y: pd.DataFrame,
    config_path: str = "config.yaml"
) -> Tuple[object, dict]:
    """
    Trains the price prediction model.
    
    Args:
        X: Features DataFrame
        y: Target DataFrame
        config_path: Path to configuration file
        
    Returns:
        Tuple of (trained model, training metrics)
    """
    config = load_config(config_path)
    model_config = get_model_config(config)
    training_config = get_training_config(config)
    data_config = config.get('features', {})
    target_col = data_config.get('target_column', 'price')
    
    # Extract target series
    y_series = y[target_col] if isinstance(y, pd.DataFrame) else y
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_series,
        test_size=training_config.test_size,
        random_state=training_config.random_state
    )
    
    print(f"📊 Training set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    
    # Select model based on configuration
    algorithm = model_config.algorithm.lower()
    
    if algorithm == "random_forest":
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=training_config.random_state,
            n_jobs=-1
        )
    elif algorithm == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=100,
            random_state=training_config.random_state
        )
    elif algorithm == "linear_regression":
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    print(f"🚀 Training {algorithm} model...")
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2_score': float(r2),
        'algorithm': algorithm,
        'model_name': model_config.name
    }
    
    print(f"✅ Model trained successfully!")
    print(f"📈 Test Metrics:")
    print(f"   MSE: {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   R² Score: {r2:.4f}")
    
    # Save model
    Path("models").mkdir(exist_ok=True)
    model_path = f"models/{model_config.name}_{model_config.version}.joblib"
    joblib.dump(model, model_path)
    print(f"💾 Model saved to {model_path}")
    
    return model, metrics

