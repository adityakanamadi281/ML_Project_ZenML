# Price Prediction ML System

A production-ready machine learning system for price prediction built with ZenML, featuring a modular architecture, automated pipelines, and comprehensive evaluation.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Price Prediction ML System                │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐          ┌─────▼─────┐         ┌─────▼─────┐
   │  Data   │          │  Feature  │         │   Model   │
   │Ingestion│─────────▶│Engineering│────────▶│ Training  │
   └─────────┘          └───────────┘         └──────────┘
        │                     │                     │
        │              ┌───────▼───────┐             │
        │              │ Preprocessing │             │
        │              └───────────────┘             │
        │                                            │
        └────────────────────┬──────────────────────┘
                             │
                      ┌──────▼───────┐
                      │  Evaluation  │
                      └──────────────┘
                             │
                      ┌──────▼───────┐
                      │  Prediction  │
                      └──────────────┘
```

## 📁 Project Structure

```
ZenML/
├── src/
│   ├── __init__.py
│   ├── schemas.py              # Data schemas and type definitions
│   ├── utils.py                # Utility functions
│   ├── steps/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py   # Data loading step
│   │   ├── data_preprocessing.py # Data cleaning step
│   │   ├── feature_engineering.py # Feature transformation step
│   │   ├── model_training.py   # Model training step
│   │   ├── model_evaluation.py # Model evaluation step
│   │   └── prediction.py       # Inference step
│   └── pipelines/
│       ├── __init__.py
│       ├── training_pipeline.py # Complete training pipeline
│       └── inference_pipeline.py # Inference pipeline
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── run_training.py            # Training script
├── run_inference.py           # Inference script
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize ZenML
zenml init
```

### 2. Prepare Your Data

Place your data file at the path specified in `config.yaml` (default: `data/raw/prices.csv`).

Your data should have:
- A target column (specified in `config.yaml` as `features.target_column`)
- Feature columns (numerical and/or categorical)

### 3. Configure the System

Edit `config.yaml` to match your data:
- Set `data.source_path` to your data file path
- Set `features.target_column` to your target variable name
- Configure `features.categorical_features` and `features.numerical_features`
- Choose your model algorithm: `random_forest`, `gradient_boosting`, or `linear_regression`

### 4. Run Training

```bash
python run_training.py
```

This will:
1. Ingest and preprocess your data
2. Engineer features (encoding, scaling)
3. Train the model
4. Evaluate and save metrics
5. Generate visualization plots

### 5. Run Inference

```bash
python run_inference.py data/new_data.csv
```

This will generate predictions and save them to `predictions.csv`.

## 🔧 Configuration

### Model Configuration

```yaml
model:
  name: "price_predictor"
  version: "1.0.0"
  algorithm: "random_forest"  # Options: random_forest, gradient_boosting, linear_regression
```

### Training Configuration

```yaml
training:
  test_size: 0.2              # Test set proportion
  validation_size: 0.1         # Validation set proportion
  random_state: 42             # Random seed for reproducibility
  cv_folds: 5                  # Cross-validation folds
```

### Feature Configuration

```yaml
features:
  target_column: "price"       # Name of target variable
  categorical_features: []     # List of categorical columns (auto-detected if empty)
  numerical_features: []       # List of numerical columns (auto-detected if empty)
  date_features: []            # List of date columns
```

## 📊 Pipeline Components

### 1. Data Ingestion (`data_ingestion.py`)
- Loads data from configured source
- Validates data structure
- Returns raw DataFrame

### 2. Data Preprocessing (`data_preprocessing.py`)
- Handles missing values (median for numerical, mode for categorical)
- Separates features and target
- Returns cleaned data

### 3. Feature Engineering (`feature_engineering.py`)
- Encodes categorical variables (Label Encoding)
- Scales numerical features (StandardScaler)
- Saves transformers for inference
- Returns engineered features

### 4. Model Training (`model_training.py`)
- Splits data into train/test sets
- Trains selected model algorithm
- Evaluates on test set
- Saves trained model
- Returns model and metrics

### 5. Model Evaluation (`model_evaluation.py`)
- Generates evaluation plots:
  - Actual vs Predicted scatter plot
  - Residual plot
- Returns comprehensive metrics

### 6. Prediction (`prediction.py`)
- Loads trained model
- Makes predictions on new data
- Returns predictions DataFrame

## 🎯 Supported Algorithms

1. **Random Forest** (`random_forest`)
   - Ensemble method, handles non-linearity well
   - Good default choice

2. **Gradient Boosting** (`gradient_boosting`)
   - Strong performance on structured data
   - Can be slower but often more accurate

3. **Linear Regression** (`linear_regression`)
   - Fast and interpretable
   - Good baseline model

## 📈 Evaluation Metrics

The system evaluates models using:
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R² Score** (Coefficient of Determination)

## 🔄 Workflow

### Training Workflow
```
Data → Preprocessing → Feature Engineering → Training → Evaluation → Model Artifacts
```

### Inference Workflow
```
New Data → Preprocessing → Feature Engineering (using saved transformers) → Prediction
```

## 📝 Outputs

After training, you'll find:
- **Models**: `models/price_predictor_1.0.0.joblib`
- **Preprocessors**: `artifacts/scaler.pkl`, `artifacts/label_encoders.pkl`
- **Plots**: `reports/plots/actual_vs_predicted.png`, `reports/plots/residuals.png`
- **Predictions**: `predictions.csv` (after inference)

## 🛠️ Extending the System

### Adding New Algorithms

1. Import your model in `src/steps/model_training.py`
2. Add a new condition in the algorithm selection logic
3. Update `config.yaml` with the new algorithm name

### Adding New Features

1. Modify `src/steps/feature_engineering.py`
2. Add your feature engineering logic
3. Update configuration if needed

### Custom Preprocessing

1. Edit `src/steps/data_preprocessing.py`
2. Add your custom preprocessing steps
3. Ensure compatibility with downstream steps

## 🐛 Troubleshooting

### Model Not Found Error
- Ensure you've run training first: `python run_training.py`
- Check that model exists at path specified in config

### Missing Columns Error
- Verify your data has all required columns
- Check `config.yaml` feature configuration

### Memory Issues
- Reduce dataset size for testing
- Use smaller models or adjust hyperparameters

## 📚 Next Steps

- Add hyperparameter tuning
- Implement model versioning
- Add data validation schemas
- Set up experiment tracking
- Deploy as API service
- Add monitoring and logging

## 🤝 Contributing

This is a template system. Customize it for your specific use case!

## 📄 License

MIT License - feel free to use and modify as needed.

