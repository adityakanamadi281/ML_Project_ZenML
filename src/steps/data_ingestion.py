"""Data ingestion step"""
import pandas as pd
from zenml import step
from typing import Tuple
from src.utils import get_data_config, load_config


@step
def ingest_data(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Ingests data from the source path.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Raw DataFrame with the ingested data
    """
    config = load_config(config_path)
    data_config = get_data_config(config)
    
    # Load data from source
    df = pd.read_csv(data_config.source_path)
    
    print(f"✅ Data ingested successfully. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

