"""Script to run the training pipeline"""
from src.pipelines.training_pipeline import training_pipeline
import sys

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    
    print("=" * 60)
    print("🚀 Starting Price Prediction ML Training Pipeline")
    print("=" * 60)
    
    # Run the pipeline
    training_pipeline(config_path=config_path)
    
    print("=" * 60)
    print("✅ Training completed!")
    print("=" * 60)

