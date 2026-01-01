"""Inference pipeline for price prediction"""
from zenml import pipeline
from zenml.logger import get_logger
from src.steps.data_preprocessing import preprocess_data
from src.steps.feature_engineering import engineer_features
from src.steps.prediction import predict

logger = get_logger(__name__)


@pipeline
def inference_pipeline(
    data,
    config_path: str = "config.yaml"
):
    """
    Inference pipeline for making predictions on new data.
    
    Pipeline steps:
    1. Preprocess input data
    2. Engineer features (using saved transformers)
    3. Make predictions
    """
    logger.info("🔮 Starting inference pipeline...")
    
    # Step 1: Preprocess data
    X, y = preprocess_data(data=data, config_path=config_path)
    
    # Step 2: Engineer features (inference mode - uses saved transformers)
    X_engineered, y_engineered, artifacts = engineer_features(
        X=X,
        y=y,
        config_path=config_path,
        is_training=False
    )
    
    # Step 3: Make predictions
    predictions = predict(
        X=X_engineered,
        config_path=config_path
    )
    
    logger.info("✅ Inference pipeline completed successfully!")
    
    return predictions

