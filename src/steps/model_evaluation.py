"""Model evaluation step"""
import pandas as pd
from zenml import step
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.utils import load_config


@step
def evaluate_model(
    model: object,
    X: pd.DataFrame,
    y: pd.DataFrame,
    metrics: dict,
    config_path: str = "config.yaml"
) -> dict:
    """
    Evaluates the model and generates visualizations.
    
    Args:
        model: Trained model
        X: Features DataFrame
        y: Target DataFrame
        metrics: Training metrics dictionary
        config_path: Path to configuration file
        
    Returns:
        Enhanced metrics dictionary
    """
    config = load_config(config_path)
    eval_config = config.get('evaluation', {})
    data_config = config.get('features', {})
    target_col = data_config.get('target_column', 'price')
    
    y_series = y[target_col] if isinstance(y, pd.DataFrame) else y
    y_pred = model.predict(X)
    
    # Generate plots if enabled
    if eval_config.get('save_plots', True):
        plots_path = Path(eval_config.get('plots_path', 'reports/plots'))
        plots_path.mkdir(parents=True, exist_ok=True)
        
        # Actual vs Predicted scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_series, y_pred, alpha=0.5)
        plt.plot([y_series.min(), y_series.max()], 
                [y_series.min(), y_series.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted Prices')
        plt.tight_layout()
        plt.savefig(plots_path / 'actual_vs_predicted.png', dpi=300)
        plt.close()
        
        # Residual plot
        residuals = y_series - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Price')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.tight_layout()
        plt.savefig(plots_path / 'residuals.png', dpi=300)
        plt.close()
        
        print(f"📊 Evaluation plots saved to {plots_path}")
    
    print("✅ Model evaluation completed!")
    
    return metrics

