# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install packages
pip install -r requirements.txt

# Initialize ZenML
zenml init
```

### Step 2: Generate Sample Data

```bash
python create_sample_data.py
```

This creates sample price prediction data at `data/raw/prices.csv` with:
- Area (sq ft)
- Bedrooms
- Bathrooms
- Age (years)
- Location (A, B, C, D)
- Condition (excellent, good, fair, poor)
- Price (target variable)

### Step 3: Run Training

```bash
python run_training.py
```

This will:
1. ✅ Load and preprocess the data
2. ✅ Engineer features (encode categorical, scale numerical)
3. ✅ Train a Random Forest model
4. ✅ Evaluate and save metrics
5. ✅ Generate visualization plots

**Expected Output:**
```
🚀 Starting Price Prediction ML Training Pipeline
✅ Data ingested successfully. Shape: (1000, 7)
🔧 Handling missing values...
🔧 Encoding 2 categorical features...
🔧 Scaling 4 numerical features...
📊 Training set: 800 samples
📊 Test set: 200 samples
🚀 Training random_forest model...
✅ Model trained successfully!
📈 Test Metrics:
   MSE: 1234567.89
   RMSE: 1111.11
   MAE: 888.88
   R² Score: 0.95
💾 Model saved to models/price_predictor_1.0.0.joblib
📊 Evaluation plots saved to reports/plots
✅ Training pipeline completed successfully!
```

### Step 4: Run Inference

```bash
# Create a small test file or use existing data
python run_inference.py data/raw/prices.csv
```

This will generate predictions and save them to `predictions.csv`.

## 📁 What Gets Created

After running the pipeline, you'll have:

```
ZenML/
├── data/
│   └── raw/
│       └── prices.csv          # Your input data
├── models/
│   └── price_predictor_1.0.0.joblib  # Trained model
├── artifacts/
│   ├── scaler.pkl              # Feature scaler
│   └── label_encoders.pkl      # Categorical encoders
├── reports/
│   └── plots/
│       ├── actual_vs_predicted.png
│       └── residuals.png
└── predictions.csv             # Inference results
```

## 🔧 Customizing for Your Data

### 1. Update Configuration

Edit `config.yaml`:

```yaml
data:
  source_path: "data/raw/your_data.csv"  # Your data path

features:
  target_column: "your_target_column"    # Your target variable
  categorical_features: ["col1", "col2"] # Your categorical columns
  numerical_features: ["col3", "col4"]    # Your numerical columns

model:
  algorithm: "gradient_boosting"  # Try different algorithms
```

### 2. Prepare Your Data

Your CSV should have:
- One column for the target (price/value to predict)
- Feature columns (can be mixed numerical and categorical)
- No missing values in target column (missing values in features are handled automatically)

### 3. Run Training

```bash
python run_training.py
```

## 🎯 Next Steps

1. **Experiment with Algorithms**
   - Change `model.algorithm` in `config.yaml`
   - Try: `random_forest`, `gradient_boosting`, `linear_regression`

2. **Improve Features**
   - Add feature engineering in `src/steps/feature_engineering.py`
   - Create derived features (e.g., price per sqft)

3. **Hyperparameter Tuning**
   - Modify model parameters in `src/steps/model_training.py`
   - Add grid search or random search

4. **Add More Metrics**
   - Extend evaluation in `src/steps/model_evaluation.py`
   - Add custom business metrics

## 🐛 Troubleshooting

### "FileNotFoundError: data/raw/prices.csv"
- Run `python create_sample_data.py` first
- Or update `config.yaml` with your data path

### "Model not found" during inference
- Run training first: `python run_training.py`
- Check that model exists in `models/` directory

### Import errors
- Make sure you've installed dependencies: `pip install -r requirements.txt`
- Activate your virtual environment

### ZenML errors
- Initialize ZenML: `zenml init`
- Check ZenML version compatibility

## 📚 Learn More

- See `README.md` for detailed documentation
- See `ARCHITECTURE.md` for system design details
- Check ZenML docs: https://docs.zenml.io

## 💡 Tips

1. **Start Small**: Test with sample data first
2. **Iterate**: Try different algorithms and features
3. **Monitor**: Check evaluation plots to understand model performance
4. **Version**: Keep track of config changes for reproducibility

Happy modeling! 🎉

