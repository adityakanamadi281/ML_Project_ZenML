# Price Prediction ML System - Architecture Design

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Price Prediction ML System                  │
│                         (ZenML Orchestration)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐          ┌─────▼─────┐         ┌─────▼─────┐
   │  Data   │          │  Feature  │         │   Model   │
   │Ingestion│─────────▶│Engineering│────────▶│ Training  │
   │  Step   │          │   Step    │         │   Step    │
   └─────────┘          └───────────┘         └──────────┘
        │                     │                     │
        │              ┌───────▼───────┐             │
        │              │ Preprocessing │             │
        │              │     Step      │             │
        │              └───────────────┘             │
        │                                            │
        └────────────────────┬──────────────────────┘
                             │
                      ┌──────▼───────┐
                      │  Evaluation  │
                      │     Step     │
                      └──────────────┘
                             │
                      ┌──────▼───────┐
                      │  Prediction  │
                      │     Step     │
                      └──────────────┘
```

## 📦 Component Design

### 1. Data Layer

#### Data Ingestion (`data_ingestion.py`)
- **Purpose**: Load raw data from source
- **Input**: Configuration file path
- **Output**: Raw pandas DataFrame
- **Responsibilities**:
  - Read data from configured source (CSV, database, API)
  - Validate data structure
  - Log data statistics

#### Data Preprocessing (`data_preprocessing.py`)
- **Purpose**: Clean and prepare data
- **Input**: Raw DataFrame
- **Output**: Features DataFrame, Target DataFrame
- **Responsibilities**:
  - Handle missing values
  - Detect and handle outliers
  - Separate features and target
  - Data type validation

### 2. Feature Engineering Layer

#### Feature Engineering (`feature_engineering.py`)
- **Purpose**: Transform features for model consumption
- **Input**: Features DataFrame, Target DataFrame
- **Output**: Engineered Features, Target, Preprocessor Artifacts
- **Responsibilities**:
  - Categorical encoding (Label Encoding)
  - Numerical scaling (StandardScaler)
  - Feature creation (if needed)
  - Save transformers for inference
- **Artifacts**:
  - `artifacts/scaler.pkl` - StandardScaler for numerical features
  - `artifacts/label_encoders.pkl` - LabelEncoders for categorical features

### 3. Model Layer

#### Model Training (`model_training.py`)
- **Purpose**: Train price prediction model
- **Input**: Engineered Features, Target
- **Output**: Trained Model, Training Metrics
- **Responsibilities**:
  - Split data (train/test)
  - Select and train model algorithm
  - Evaluate on test set
  - Save trained model
- **Supported Algorithms**:
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Linear Regression
- **Output Artifacts**:
  - `models/{model_name}_{version}.joblib` - Trained model

#### Model Evaluation (`model_evaluation.py`)
- **Purpose**: Comprehensive model evaluation
- **Input**: Trained Model, Features, Target, Metrics
- **Output**: Enhanced Metrics Dictionary
- **Responsibilities**:
  - Generate evaluation plots
  - Calculate additional metrics
  - Visualize predictions vs actuals
  - Residual analysis
- **Output Artifacts**:
  - `reports/plots/actual_vs_predicted.png`
  - `reports/plots/residuals.png`

### 4. Inference Layer

#### Prediction (`prediction.py`)
- **Purpose**: Make predictions on new data
- **Input**: Features DataFrame
- **Output**: Predictions DataFrame
- **Responsibilities**:
  - Load trained model
  - Apply feature transformations
  - Generate predictions
  - Format output

## 🔄 Pipeline Flows

### Training Pipeline Flow

```
┌──────────────┐
│ Config File  │
└──────┬───────┘
       │
       ▼
┌─────────────────┐
│ Data Ingestion  │ ──▶ Raw DataFrame
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│ Preprocessing    │ ──▶ Cleaned Features + Target
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Feature          │ ──▶ Engineered Features + Artifacts
│ Engineering      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Model Training   │ ──▶ Trained Model + Metrics
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Evaluation       │ ──▶ Final Metrics + Plots
└──────────────────┘
```

### Inference Pipeline Flow

```
┌──────────────┐
│ New Data     │
└──────┬───────┘
       │
       ▼
┌─────────────────┐
│ Preprocessing   │ ──▶ Cleaned Features
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│ Feature          │ ──▶ Engineered Features
│ Engineering      │     (using saved transformers)
│ (Inference Mode) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Prediction      │ ──▶ Predictions DataFrame
└──────────────────┘
```

## 🗂️ Data Flow

### Training Phase

```
Raw Data (CSV)
    │
    ├─▶ Data Ingestion
    │       │
    │       └─▶ Raw DataFrame
    │
    ├─▶ Preprocessing
    │       │
    │       ├─▶ Features DataFrame
    │       └─▶ Target DataFrame
    │
    ├─▶ Feature Engineering
    │       │
    │       ├─▶ Engineered Features
    │       ├─▶ Scaler (saved)
    │       └─▶ Encoders (saved)
    │
    ├─▶ Model Training
    │       │
    │       ├─▶ Trained Model (saved)
    │       └─▶ Training Metrics
    │
    └─▶ Evaluation
            │
            ├─▶ Evaluation Metrics
            └─▶ Visualization Plots
```

### Inference Phase

```
New Data (CSV)
    │
    ├─▶ Preprocessing
    │       │
    │       └─▶ Cleaned Features
    │
    ├─▶ Feature Engineering
    │       │
    │       ├─▶ Load Scaler
    │       ├─▶ Load Encoders
    │       └─▶ Engineered Features
    │
    └─▶ Prediction
            │
            ├─▶ Load Model
            └─▶ Predictions (CSV)
```

## 🔧 Configuration Management

### Configuration Structure

```yaml
data:
  source_path: "data/raw/prices.csv"
  processed_path: "data/processed/"
  features_path: "data/features/"

model:
  name: "price_predictor"
  version: "1.0.0"
  algorithm: "random_forest"

training:
  test_size: 0.2
  validation_size: 0.1
  random_state: 42
  cv_folds: 5

features:
  target_column: "price"
  categorical_features: []
  numerical_features: []
  date_features: []

evaluation:
  metrics: ["mse", "rmse", "mae", "r2_score"]
  save_plots: true
  plots_path: "reports/plots/"
```

### Configuration Benefits

- **Centralized**: All settings in one place
- **Versioned**: Track configuration changes
- **Flexible**: Easy to switch algorithms, adjust parameters
- **Reproducible**: Same config = same results

## 📊 Artifact Management

### Artifacts Produced

1. **Models**
   - Location: `models/{model_name}_{version}.joblib`
   - Format: Joblib serialized model
   - Versioned: Yes (via version in config)

2. **Preprocessors**
   - Location: `artifacts/scaler.pkl`, `artifacts/label_encoders.pkl`
   - Format: Pickle serialized transformers
   - Purpose: Ensure consistent transformations in inference

3. **Visualizations**
   - Location: `reports/plots/`
   - Formats: PNG images
   - Types: Actual vs Predicted, Residual plots

4. **Predictions**
   - Location: `predictions.csv`
   - Format: CSV with predicted prices

## 🎯 Design Principles

### 1. Modularity
- Each step is independent and testable
- Easy to swap components
- Clear separation of concerns

### 2. Reproducibility
- Fixed random seeds
- Versioned models and configs
- Saved transformers ensure consistent preprocessing

### 3. Scalability
- ZenML handles orchestration
- Can easily add new steps
- Supports distributed execution

### 4. Maintainability
- Clear code structure
- Comprehensive documentation
- Type hints and schemas

### 5. Extensibility
- Easy to add new algorithms
- Simple to add new features
- Configurable without code changes

## 🔄 Extension Points

### Adding New Algorithms

1. Import model in `model_training.py`
2. Add condition in algorithm selection
3. Update config with new algorithm name

### Adding New Preprocessing Steps

1. Add step in `data_preprocessing.py`
2. Ensure output format compatibility
3. Update documentation

### Adding New Features

1. Modify `feature_engineering.py`
2. Add feature creation logic
3. Update config if needed

### Adding Monitoring

1. Integrate with ZenML experiment tracking
2. Add logging to each step
3. Set up model performance monitoring

## 🚀 Deployment Considerations

### Current Architecture Supports

- ✅ Local development and testing
- ✅ Model versioning
- ✅ Reproducible pipelines
- ✅ Batch inference

### Future Enhancements

- 🔄 Real-time inference API
- 🔄 Model serving (MLflow, Seldon)
- 🔄 A/B testing framework
- 🔄 Automated retraining
- 🔄 Data drift detection
- 🔄 Model monitoring dashboard

## 📈 Performance Considerations

### Training
- Model selection based on data size
- Cross-validation for robust evaluation
- Feature importance analysis for optimization

### Inference
- Cached transformers (loaded once)
- Efficient model loading
- Batch processing support

## 🔒 Best Practices Implemented

1. **Data Validation**: Schema validation with Pydantic
2. **Error Handling**: Graceful error messages
3. **Logging**: Comprehensive logging at each step
4. **Documentation**: Inline and external docs
5. **Version Control**: Model and config versioning
6. **Testing Ready**: Modular design enables unit testing

