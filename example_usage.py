"""
Example usage of Well Performance Predictor
This file demonstrates how to use the WellPerformancePredictor with your actual data
"""

from well_performance_predictor import WellPerformancePredictor
import pandas as pd
import numpy as np

def example_with_csv_files():
    """
    Example using CSV files
    """
    print("=== Example 1: Using CSV Files ===")
    
    # Initialize predictor
    predictor = WellPerformancePredictor()
    
    # Load data from CSV files
    # Replace with your actual file paths
    predictor.load_operational_data(file_path='operational_data.csv')
    predictor.load_test_data(file_path='test_data.csv')
    
    # Train and predict
    features = predictor.engineer_features()
    targets = predictor.prepare_targets()
    results = predictor.train_models(targets)
    
    # Generate report and make predictions
    predictor.generate_report(results)
    predictions = predictor.predict()
    print("\nPredictions:", predictions)

def example_with_dictionaries():
    """
    Example using data dictionaries (your format)
    """
    print("=== Example 2: Using Data Dictionaries ===")
    
    # Example operational data (2 weeks, 2-hour intervals)
    operational_data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=168, freq='2H'),
        'thp': [1450, 1460, 1455, 1470, 1465] * 34,  # Tubing Head Pressure (psi)
        'downstring_pressure': [2100, 2110, 2105, 2120, 2115] * 34,  # Downstring pressure (psi)
        'temperature': [175, 180, 178, 185, 182] * 34,  # Temperature (°F)
        'choke_size': [48, 48, 56, 56, 64] * 34  # Choke size (64ths)
    }
    
    # Example test data (last 10 tests)
    test_data = {
        'test_date': pd.date_range(start='2023-12-01', periods=10, freq='7D'),
        'oil_rate': [950, 980, 1020, 990, 1050, 1030, 1080, 1060, 1100, 1090],  # bbl/day
        'gas_rate': [1800, 1850, 1900, 1820, 1950, 1880, 2000, 1920, 2050, 2020],  # Mcf/day
        'water_cut': [15, 18, 20, 22, 25, 28, 30, 32, 35, 38],  # %
        'total_liquid': [1100, 1150, 1200, 1180, 1250, 1220, 1300, 1280, 1350, 1320]  # bbl/day
    }
    
    # Initialize and load data
    predictor = WellPerformancePredictor()
    predictor.load_operational_data(data_dict=operational_data)
    predictor.load_test_data(data_dict=test_data)
    
    # Train models
    features = predictor.engineer_features()
    print(f"Created {len(features.columns)} engineered features")
    
    targets = predictor.prepare_targets()
    results = predictor.train_models(targets)
    
    # Generate comprehensive report
    predictor.generate_report(results)
    
    # Make predictions with current conditions
    current_predictions = predictor.predict()
    print("\nCurrent Well Performance Predictions:")
    print(f"Oil Rate: {current_predictions['oil_rate']:.2f} bbl/day")
    print(f"Gas Rate: {current_predictions['gas_rate']:.2f} Mcf/day")
    print(f"Water Cut: {current_predictions['water_cut']:.2f}%")
    print(f"Total Liquid: {current_predictions['total_liquid']:.2f} bbl/day")
    
    # Make predictions with new operational conditions
    new_conditions = {
        'thp': 1520,
        'downstring_pressure': 2150,
        'temperature': 190,
        'choke_size': 72,
        # Add all other required features (the system will handle this)
    }
    
    # For new conditions, we need to create a full feature set
    # The predictor will use the latest engineered features as a template
    print("\nPredictions with new choke size (72/64ths):")
    # Note: For new conditions, you'd typically adjust the latest operational data
    # and re-engineer features for the most accurate prediction
    
    return predictor, results

def example_with_real_time_prediction():
    """
    Example for real-time prediction with new data points
    """
    print("=== Example 3: Real-time Prediction ===")
    
    # Assume you have a trained model
    predictor = WellPerformancePredictor()
    
    # Load historical data and train
    predictor.load_operational_data()  # Uses sample data
    predictor.load_test_data()  # Uses sample data
    
    features = predictor.engineer_features()
    targets = predictor.prepare_targets()
    results = predictor.train_models(targets)
    
    # Save the trained model
    predictor.save_models('trained_well_model.pkl')
    
    # Later, load the model for predictions
    new_predictor = WellPerformancePredictor()
    new_predictor.load_models('trained_well_model.pkl')
    
    # Make predictions with new operational data
    # (This would be your live data feed)
    new_data_point = {
        'thp': 1480,
        'downstring_pressure': 2080,
        'temperature': 185,
        'choke_size': 60,
        # Additional features will be engineered automatically
    }
    
    print("Model loaded successfully for real-time predictions")

def create_sample_csv_files():
    """
    Create sample CSV files in the expected format
    """
    print("Creating sample CSV files...")
    
    # Operational data CSV
    dates = pd.date_range(start='2024-01-01', periods=168, freq='2H')
    operational_df = pd.DataFrame({
        'timestamp': dates,
        'thp': np.random.normal(1500, 100, len(dates)),
        'downstring_pressure': np.random.normal(2000, 150, len(dates)),
        'temperature': np.random.normal(180, 20, len(dates)),
        'choke_size': np.random.choice([32, 40, 48, 56, 64], len(dates))
    })
    operational_df.to_csv('operational_data.csv', index=False)
    
    # Test data CSV
    test_dates = pd.date_range(start='2023-12-01', periods=10, freq='7D')
    test_df = pd.DataFrame({
        'test_date': test_dates,
        'oil_rate': np.random.normal(1000, 200, 10),
        'gas_rate': np.random.normal(2000, 400, 10),
        'water_cut': np.random.uniform(10, 30, 10),
        'total_liquid': np.random.normal(1200, 250, 10)
    })
    test_df.to_csv('test_data.csv', index=False)
    
    print("Sample CSV files created: operational_data.csv, test_data.csv")
    print("\nOperational data format:")
    print(operational_df.head())
    print("\nTest data format:")
    print(test_df.head())

def analyze_feature_importance():
    """
    Example focusing on feature importance analysis
    """
    print("=== Feature Importance Analysis ===")
    
    predictor = WellPerformancePredictor()
    predictor.load_operational_data()
    predictor.load_test_data()
    
    features = predictor.engineer_features()
    targets = predictor.prepare_targets()
    results = predictor.train_models(targets)
    
    # Analyze what features are most important for each target
    for target in ['oil_rate', 'gas_rate', 'water_cut', 'total_liquid']:
        if target in predictor.feature_importance:
            print(f"\nTop 10 features for {target}:")
            importance_dict = predictor.feature_importance[target]
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            for feature, importance in sorted_features:
                print(f"  {feature}: {importance:.4f}")

if __name__ == "__main__":
    # Run different examples
    print("Well Performance Predictor - Usage Examples")
    print("=" * 50)
    
    # Create sample files first
    create_sample_csv_files()
    
    # Run main example
    predictor, results = example_with_dictionaries()
    
    # Analyze feature importance
    analyze_feature_importance()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo use with your data:")
    print("1. Format your data according to the examples above")
    print("2. Use load_operational_data() and load_test_data() methods")
    print("3. Call train_models() to train")
    print("4. Use predict() for new predictions")
    print("5. Save/load models with save_models() and load_models()")