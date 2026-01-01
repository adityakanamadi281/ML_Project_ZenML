"""Script to run inference on new data"""
import pandas as pd
import sys
from src.pipelines.inference_pipeline import inference_pipeline

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_inference.py <path_to_data.csv> [config.yaml]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
    
    print("=" * 60)
    print("🔮 Starting Price Prediction Inference")
    print("=" * 60)
    
    # Load data
    data = pd.read_csv(data_path)
    print(f"📊 Loaded {len(data)} samples for prediction")
    
    # Run inference pipeline
    predictions = inference_pipeline(data=data, config_path=config_path)
    
    # Save predictions
    output_path = "predictions.csv"
    predictions.to_csv(output_path, index=False)
    
    print(f"✅ Predictions saved to {output_path}")
    print("=" * 60)

