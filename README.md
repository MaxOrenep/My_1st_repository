# Well Performance Predictor

A comprehensive machine learning system for predicting well performance using operational data and historical test results.

## Features

- **Multi-target Prediction**: Predicts oil rate, gas rate, water cut, and total liquid production
- **Advanced Feature Engineering**: Creates over 40 engineered features from raw operational data
- **Multiple ML Models**: Compares Linear Regression, Ridge Regression, Random Forest, and Gradient Boosting
- **Time Series Analysis**: Incorporates rolling statistics, lag features, and rate of change
- **Model Persistence**: Save and load trained models for production use
- **Comprehensive Reporting**: Detailed performance metrics and feature importance analysis
- **Visualization**: Plots for model performance and feature importance

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the example:
```bash
python well_performance_predictor.py
```

## Data Format

### Operational Data (2-week history, 2-hour intervals)
```python
operational_data = {
    'timestamp': datetime_series,
    'thp': [1450, 1460, ...],              # Tubing Head Pressure (psi)
    'downstring_pressure': [2100, 2110, ...],  # Downstring pressure (psi)
    'temperature': [175, 180, ...],         # Temperature (°F)
    'choke_size': [48, 56, ...]            # Choke size (64ths)
}
```

### Test Data (Last 10 well tests)
```python
test_data = {
    'test_date': datetime_series,
    'oil_rate': [950, 980, ...],           # Oil production (bbl/day)
    'gas_rate': [1800, 1850, ...],         # Gas production (Mcf/day)
    'water_cut': [15, 18, ...],            # Water cut (%)
    'total_liquid': [1100, 1150, ...]      # Total liquid (bbl/day)
}
```

## Quick Start

```python
from well_performance_predictor import WellPerformancePredictor

# Initialize predictor
predictor = WellPerformancePredictor()

# Load your data
predictor.load_operational_data(data_dict=your_operational_data)
predictor.load_test_data(data_dict=your_test_data)

# Train models
features = predictor.engineer_features()
targets = predictor.prepare_targets()
results = predictor.train_models(targets)

# Make predictions
predictions = predictor.predict()
print(f"Oil Rate: {predictions['oil_rate']:.2f} bbl/day")
print(f"Gas Rate: {predictions['gas_rate']:.2f} Mcf/day")
print(f"Water Cut: {predictions['water_cut']:.2f}%")
print(f"Total Liquid: {predictions['total_liquid']:.2f} bbl/day")

# Save model for later use
predictor.save_models('my_well_model.pkl')
```

## Advanced Usage

### Loading from CSV Files
```python
predictor.load_operational_data(file_path='operational_data.csv')
predictor.load_test_data(file_path='test_data.csv')
```

### Feature Importance Analysis
```python
predictor.plot_feature_importance('oil_rate', top_n=15)
```

### Model Performance Visualization
```python
predictor.plot_predictions_vs_actual(results, 'oil_rate')
```

### Real-time Predictions
```python
# Load pre-trained model
predictor.load_models('trained_model.pkl')

# Make prediction with new data
new_prediction = predictor.predict(new_operational_data)
```

## Engineered Features

The system automatically creates advanced features including:

- **Rolling Statistics**: 24-hour moving averages, standard deviations, min/max
- **Pressure Analysis**: Pressure differentials and ratios
- **Temporal Features**: Hour, day of week, day of month
- **Lag Features**: Previous 1-2 time step values
- **Rate of Change**: First derivatives of key parameters
- **Choke Performance**: Pressure-to-choke ratios
- **Historical Trends**: Test result trends and averages

## Model Selection

The system automatically selects the best performing model for each target based on R² score:

- **Linear Regression**: Fast, interpretable baseline
- **Ridge Regression**: Regularized linear model
- **Random Forest**: Ensemble method, handles non-linearity
- **Gradient Boosting**: Advanced ensemble, often best performance

## Performance Metrics

- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Feature Importance**: For tree-based models

## Example Output

```
WELL PERFORMANCE PREDICTION REPORT
============================================================

OIL_RATE PREDICTION:
----------------------------------------
Best Model: gradient_boosting
R² Score: 0.8756
RMSE: 45.23
MAE: 32.18

Top 5 Important Features:
  latest_oil_rate: 0.2134
  thp_rolling_mean_24h: 0.1523
  pressure_diff: 0.1287
  choke_pressure_ratio: 0.0945
  temperature_rolling_std_24h: 0.0834
```

## File Structure

```
├── well_performance_predictor.py  # Main prediction system
├── example_usage.py              # Usage examples
├── requirements.txt              # Dependencies
├── README.md                    # This file
└── models/                      # Saved models (created after training)
```

## Customization

### Adding New Features
Extend the `engineer_features()` method to add domain-specific features:

```python
def engineer_features(self):
    # ... existing code ...
    
    # Add custom features
    op_features['custom_ratio'] = op_features['thp'] / op_features['temperature']
    op_features['production_index'] = op_features['choke_size'] * op_features['pressure_diff']
    
    # ... rest of method ...
```

### Adding New Models
Extend the models dictionary in `train_models()`:

```python
models = {
    'linear_regression': LinearRegression(),
    'ridge_regression': Ridge(alpha=1.0),
    'random_forest': RandomForestRegressor(n_estimators=100),
    'gradient_boosting': GradientBoostingRegressor(n_estimators=100),
    'xgboost': XGBRegressor(n_estimators=100)  # Add new model
}
```

## Tips for Best Results

1. **Data Quality**: Ensure consistent, clean data with minimal gaps
2. **Feature Engineering**: Add domain-specific features based on your well characteristics
3. **Model Tuning**: Use GridSearchCV for hyperparameter optimization
4. **Regular Retraining**: Update models with new test data
5. **Validation**: Cross-validate on multiple wells if available

## Troubleshooting

### Common Issues

1. **Mismatched Data Lengths**: Ensure operational and test data time ranges align
2. **Missing Values**: The system handles missing values automatically, but clean data performs better
3. **Feature Scaling**: Linear models use automatic scaling; tree models use raw features
4. **Memory Usage**: For large datasets, consider data chunking or feature selection

### Performance Optimization

- Use feature selection for high-dimensional data
- Consider time-based cross-validation for time series
- Implement early stopping for gradient boosting models
- Use parallel processing for model training

## Contributing

To extend this system:

1. Add new feature engineering methods
2. Implement additional ML models
3. Create specialized visualization functions
4. Add domain-specific validation rules

## License

This project is provided as-is for educational and research purposes.
