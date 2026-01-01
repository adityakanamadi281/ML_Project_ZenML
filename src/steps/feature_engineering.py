"""Feature engineering step"""
import pandas as pd
from zenml import step
from typing import Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.utils import get_data_config, load_config
import pickle
from pathlib import Path


@step
def engineer_features(
    X: pd.DataFrame,
    y: pd.DataFrame,
    config_path: str = "config.yaml",
    is_training: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Engineers features: encoding, scaling, feature creation.
    
    Args:
        X: Features DataFrame
        y: Target DataFrame
        config_path: Path to configuration file
        is_training: Whether this is training phase (fit transformers) or inference
        
    Returns:
        Tuple of (engineered features, target, preprocessor artifacts)
    """
    config = load_config(config_path)
    data_config = get_data_config(config)
    
    X_processed = X.copy()
    artifacts = {}
    
    # Handle categorical features
    categorical_cols = data_config.categorical_features
    if not categorical_cols:
        # Auto-detect categorical columns
        categorical_cols = X_processed.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        print(f"🔧 Encoding {len(categorical_cols)} categorical features...")
        if is_training:
            label_encoders = {}
            for col in categorical_cols:
                if col in X_processed.columns:
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                    label_encoders[col] = le
            artifacts['label_encoders'] = label_encoders
        else:
            # Load encoders for inference
            encoders_path = Path("artifacts/label_encoders.pkl")
            if encoders_path.exists():
                with open(encoders_path, 'rb') as f:
                    label_encoders = pickle.load(f)
                for col in categorical_cols:
                    if col in X_processed.columns and col in label_encoders:
                        le = label_encoders[col]
                        # Handle unseen categories
                        X_processed[col] = X_processed[col].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
    
    # Handle numerical features - scaling
    numerical_cols = data_config.numerical_features
    if not numerical_cols:
        numerical_cols = X_processed.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if numerical_cols:
        print(f"🔧 Scaling {len(numerical_cols)} numerical features...")
        if is_training:
            scaler = StandardScaler()
            X_processed[numerical_cols] = scaler.fit_transform(X_processed[numerical_cols])
            artifacts['scaler'] = scaler
        else:
            # Load scaler for inference
            scaler_path = Path("artifacts/scaler.pkl")
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                X_processed[numerical_cols] = scaler.transform(X_processed[numerical_cols])
    
    # Save artifacts if training
    if is_training and artifacts:
        Path("artifacts").mkdir(exist_ok=True)
        if 'label_encoders' in artifacts:
            with open("artifacts/label_encoders.pkl", 'wb') as f:
                pickle.dump(artifacts['label_encoders'], f)
        if 'scaler' in artifacts:
            with open("artifacts/scaler.pkl", 'wb') as f:
                pickle.dump(artifacts['scaler'], f)
    
    print(f"✅ Features engineered. Final shape: {X_processed.shape}")
    
    return X_processed, y, artifacts

