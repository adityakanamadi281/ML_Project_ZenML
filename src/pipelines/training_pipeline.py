"""Training pipeline for price prediction"""
from zenml import pipeline
from zenml.logger import get_logger
from src.steps.data_ingestion import ingest_data
from src.steps.data_preprocessing import preprocess_data
from src.steps.feature_engineering import engineer_features
from src.steps.model_training import train_model
from src.steps.model_evaluation import evaluate_model

logger = get_logger(__name__)


@pipeline
def training_pipeline(config_path: str = "config.yaml"):
    """
    Complete training pipeline for price prediction.
    
    Pipeline steps:
    1. Ingest data from source
    2. Preprocess data (handle missing values, etc.)
    3. Engineer features (encoding, scaling)
    4. Train model
    5. Evaluate model
    """
    logger.info("🚀 Starting training pipeline...")
    
    # Step 1: Ingest data
    raw_data = ingest_data(config_path=config_path)
    
    # Step 2: Preprocess data
    X, y = preprocess_data(data=raw_data, config_path=config_path)
    
    # Step 3: Engineer features
    X_engineered, y_engineered, artifacts = engineer_features(
        X=X, 
        y=y, 
        config_path=config_path,
        is_training=True
    )
    
    # Step 4: Train model
    model, metrics = train_model(
        X=X_engineered,
        y=y_engineered,
        config_path=config_path
    )
    
    # Step 5: Evaluate model
    evaluation_metrics = evaluate_model(
        model=model,
        X=X_engineered,
        y=y_engineered,
        metrics=metrics,
        config_path=config_path
    )
    
    logger.info("✅ Training pipeline completed successfully!")
    
    return evaluation_metrics

