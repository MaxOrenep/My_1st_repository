import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class WellPerformancePredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.trained_targets = []
        
    def load_operational_data(self, file_path=None, data_dict=None):
        """
        Load 2-week operational data (THP, downstring pressure, temperature, choke size)
        
        Expected format:
        - timestamp: datetime
        - thp: Tubing Head Pressure (psi)
        - downstring_pressure: Downstring pressure (psi)
        - temperature: Temperature (°F or °C)
        - choke_size: Choke size (in 2-hour intervals)
        """
        if data_dict is not None:
            self.operational_data = pd.DataFrame(data_dict)
        elif file_path is not None:
            self.operational_data = pd.read_csv(file_path)
        else:
            # Create sample data structure
            print("Creating sample operational data structure...")
            dates = pd.date_range(start='2024-01-01', periods=168, freq='2H')  # 2 weeks, 2-hour intervals
            self.operational_data = pd.DataFrame({
                'timestamp': dates,
                'thp': np.random.normal(1500, 100, len(dates)),  # THP in psi
                'downstring_pressure': np.random.normal(2000, 150, len(dates)),  # Pressure in psi
                'temperature': np.random.normal(180, 20, len(dates)),  # Temperature in °F
                'choke_size': np.random.choice([32, 40, 48, 56, 64], len(dates))  # Choke size in 64ths
            })
            
        self.operational_data['timestamp'] = pd.to_datetime(self.operational_data['timestamp'])
        return self.operational_data
    
    def load_test_data(self, file_path=None, data_dict=None):
        """
        Load well test results (last 10 tests)
        
        Expected format:
        - test_date: datetime
        - oil_rate: Oil production rate (bbl/day)
        - gas_rate: Gas production rate (Mcf/day)
        - water_cut: Water cut percentage (%)
        - total_liquid: Total liquid rate (bbl/day)
        """
        if data_dict is not None:
            self.test_data = pd.DataFrame(data_dict)
        elif file_path is not None:
            self.test_data = pd.read_csv(file_path)
        else:
            # Create sample test data
            print("Creating sample test data structure...")
            test_dates = pd.date_range(start='2023-12-01', periods=10, freq='7D')  # 10 tests, weekly
            self.test_data = pd.DataFrame({
                'test_date': test_dates,
                'oil_rate': np.random.normal(1000, 200, 10),  # bbl/day
                'gas_rate': np.random.normal(2000, 400, 10),  # Mcf/day
                'water_cut': np.random.uniform(10, 30, 10),  # %
                'total_liquid': np.random.normal(1200, 250, 10)  # bbl/day
            })
            
        self.test_data['test_date'] = pd.to_datetime(self.test_data['test_date'])
        return self.test_data
    
    def engineer_features(self):
        """
        Create engineered features from operational and test data
        """
        # Operational data features
        op_features = self.operational_data.copy()
        
        # Time-based features
        op_features['hour'] = op_features['timestamp'].dt.hour
        op_features['day_of_week'] = op_features['timestamp'].dt.dayofweek
        op_features['day_of_month'] = op_features['timestamp'].dt.day
        
        # Rolling statistics (last 24 hours = 12 data points)
        for col in ['thp', 'downstring_pressure', 'temperature']:
            op_features[f'{col}_rolling_mean_24h'] = op_features[col].rolling(window=12, min_periods=1).mean()
            op_features[f'{col}_rolling_std_24h'] = op_features[col].rolling(window=12, min_periods=1).std()
            op_features[f'{col}_rolling_max_24h'] = op_features[col].rolling(window=12, min_periods=1).max()
            op_features[f'{col}_rolling_min_24h'] = op_features[col].rolling(window=12, min_periods=1).min()
        
        # Pressure differentials
        op_features['pressure_diff'] = op_features['downstring_pressure'] - op_features['thp']
        op_features['pressure_ratio'] = op_features['thp'] / op_features['downstring_pressure']
        
        # Choke performance metrics
        op_features['choke_pressure_ratio'] = op_features['thp'] / op_features['choke_size']
        
        # Lag features
        for col in ['thp', 'downstring_pressure', 'temperature', 'choke_size']:
            op_features[f'{col}_lag1'] = op_features[col].shift(1)
            op_features[f'{col}_lag2'] = op_features[col].shift(2)
        
        # Rate of change
        for col in ['thp', 'downstring_pressure', 'temperature']:
            op_features[f'{col}_rate_change'] = op_features[col].diff()
        
        # Test data features (latest test results)
        if hasattr(self, 'test_data') and not self.test_data.empty:
            latest_test = self.test_data.iloc[-1]
            op_features['latest_oil_rate'] = latest_test['oil_rate']
            op_features['latest_gas_rate'] = latest_test['gas_rate']
            op_features['latest_water_cut'] = latest_test['water_cut']
            op_features['latest_total_liquid'] = latest_test['total_liquid']
            
            # Historical averages
            op_features['avg_oil_rate'] = self.test_data['oil_rate'].mean()
            op_features['avg_gas_rate'] = self.test_data['gas_rate'].mean()
            op_features['avg_water_cut'] = self.test_data['water_cut'].mean()
            op_features['avg_total_liquid'] = self.test_data['total_liquid'].mean()
            
            # Trends (last 3 tests vs previous)
            if len(self.test_data) >= 6:
                recent_avg = self.test_data.tail(3)[['oil_rate', 'gas_rate', 'water_cut', 'total_liquid']].mean()
                previous_avg = self.test_data.iloc[-6:-3][['oil_rate', 'gas_rate', 'water_cut', 'total_liquid']].mean()
                
                op_features['oil_rate_trend'] = (recent_avg['oil_rate'] - previous_avg['oil_rate']) / previous_avg['oil_rate']
                op_features['gas_rate_trend'] = (recent_avg['gas_rate'] - previous_avg['gas_rate']) / previous_avg['gas_rate']
                op_features['water_cut_trend'] = recent_avg['water_cut'] - previous_avg['water_cut']
                op_features['total_liquid_trend'] = (recent_avg['total_liquid'] - previous_avg['total_liquid']) / previous_avg['total_liquid']
        
        # Remove timestamp and handle missing values
        feature_cols = [col for col in op_features.columns if col != 'timestamp']
        self.features = op_features[feature_cols].fillna(method='ffill').fillna(method='bfill')
        
        return self.features
    
    def prepare_targets(self):
        """
        Prepare target variables for prediction
        """
        if not hasattr(self, 'test_data'):
            raise ValueError("Test data not loaded. Please load test data first.")
        
        # Create targets based on test data
        # For this example, we'll predict the next test results
        targets = {
            'oil_rate': self.test_data['oil_rate'].values,
            'gas_rate': self.test_data['gas_rate'].values,
            'water_cut': self.test_data['water_cut'].values,
            'total_liquid': self.test_data['total_liquid'].values
        }
        
        return targets
    
    def train_models(self, targets_dict, test_size=0.2, random_state=42):
        """
        Train multiple models for each target variable
        """
        if not hasattr(self, 'features'):
            self.engineer_features()
        
        # Prepare features for the same time period as test data
        # For simplicity, we'll use the latest operational data
        X = self.features.tail(len(list(targets_dict.values())[0]))
        
        results = {}
        
        for target_name, y in targets_dict.items():
            print(f"\nTraining models for {target_name}...")
            
            if len(X) != len(y):
                print(f"Warning: Feature and target lengths don't match for {target_name}")
                min_len = min(len(X), len(y))
                X_target = X.tail(min_len)
                y_target = y[-min_len:]
            else:
                X_target = X
                y_target = y
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_target, y_target, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Define models
            models = {
                'linear_regression': LinearRegression(),
                'ridge_regression': Ridge(alpha=1.0),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state)
            }
            
            target_results = {}
            best_score = -np.inf
            best_model_name = None
            
            for model_name, model in models.items():
                # Train model
                if model_name in ['linear_regression', 'ridge_regression']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Evaluate
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                target_results[model_name] = {
                    'model': model,
                    'mse': mse,
                    'r2': r2,
                    'mae': mae,
                    'predictions': y_pred,
                    'actual': y_test
                }
                
                print(f"{model_name}: R² = {r2:.4f}, MSE = {mse:.4f}, MAE = {mae:.4f}")
                
                if r2 > best_score:
                    best_score = r2
                    best_model_name = model_name
            
            # Store best model and scaler
            self.models[target_name] = target_results[best_model_name]['model']
            self.scalers[target_name] = scaler if best_model_name in ['linear_regression', 'ridge_regression'] else None
            
            # Feature importance for tree-based models
            if best_model_name in ['random_forest', 'gradient_boosting']:
                importance = target_results[best_model_name]['model'].feature_importances_
                feature_names = X_target.columns
                self.feature_importance[target_name] = dict(zip(feature_names, importance))
            
            results[target_name] = target_results
            self.trained_targets.append(target_name)
            print(f"Best model for {target_name}: {best_model_name} (R² = {best_score:.4f})")
        
        return results
    
    def predict(self, new_operational_data=None):
        """
        Make predictions using trained models
        """
        if not self.models:
            raise ValueError("No trained models found. Please train models first.")
        
        if new_operational_data is not None:
            # Use provided data
            if isinstance(new_operational_data, dict):
                new_data = pd.DataFrame([new_operational_data])
            else:
                new_data = new_operational_data
        else:
            # Use latest operational data
            new_data = self.features.tail(1)
        
        predictions = {}
        
        for target_name in self.trained_targets:
            model = self.models[target_name]
            scaler = self.scalers[target_name]
            
            if scaler is not None:
                X_pred = scaler.transform(new_data)
            else:
                X_pred = new_data
            
            pred = model.predict(X_pred)[0]
            predictions[target_name] = pred
        
        return predictions
    
    def plot_feature_importance(self, target_name, top_n=15):
        """
        Plot feature importance for a specific target
        """
        if target_name not in self.feature_importance:
            print(f"Feature importance not available for {target_name}")
            return
        
        importance_dict = self.feature_importance[target_name]
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        features, importance = zip(*sorted_features)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), importance)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Features for {target_name} Prediction')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def plot_predictions_vs_actual(self, results, target_name):
        """
        Plot predictions vs actual values
        """
        best_model_key = max(results[target_name].keys(), 
                           key=lambda k: results[target_name][k]['r2'])
        
        predictions = results[target_name][best_model_key]['predictions']
        actual = results[target_name][best_model_key]['actual']
        
        plt.figure(figsize=(8, 6))
        plt.scatter(actual, predictions, alpha=0.7)
        plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{target_name} - Predictions vs Actual ({best_model_key})')
        
        # Add R² score
        r2 = results[target_name][best_model_key]['r2']
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, results):
        """
        Generate a comprehensive performance report
        """
        print("\n" + "="*60)
        print("WELL PERFORMANCE PREDICTION REPORT")
        print("="*60)
        
        for target_name, target_results in results.items():
            print(f"\n{target_name.upper()} PREDICTION:")
            print("-" * 40)
            
            best_model_key = max(target_results.keys(), 
                               key=lambda k: target_results[k]['r2'])
            best_results = target_results[best_model_key]
            
            print(f"Best Model: {best_model_key}")
            print(f"R² Score: {best_results['r2']:.4f}")
            print(f"RMSE: {np.sqrt(best_results['mse']):.4f}")
            print(f"MAE: {best_results['mae']:.4f}")
            
            if target_name in self.feature_importance:
                top_features = sorted(self.feature_importance[target_name].items(), 
                                    key=lambda x: x[1], reverse=True)[:5]
                print("\nTop 5 Important Features:")
                for feature, importance in top_features:
                    print(f"  {feature}: {importance:.4f}")
    
    def save_models(self, filepath):
        """
        Save trained models and scalers
        """
        import pickle
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_importance': self.feature_importance,
            'trained_targets': self.trained_targets,
            'feature_columns': self.features.columns.tolist() if hasattr(self, 'features') else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath):
        """
        Load trained models and scalers
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_importance = model_data['feature_importance']
        self.trained_targets = model_data['trained_targets']
        
        print(f"Models loaded from {filepath}")


def main():
    """
    Example usage of the WellPerformancePredictor
    """
    # Initialize predictor
    predictor = WellPerformancePredictor()
    
    # Load data (using sample data for demonstration)
    print("Loading operational data...")
    predictor.load_operational_data()
    
    print("Loading test data...")
    predictor.load_test_data()
    
    # Engineer features
    print("Engineering features...")
    features = predictor.engineer_features()
    print(f"Created {len(features.columns)} features")
    
    # Prepare targets
    targets = predictor.prepare_targets()
    
    # Train models
    print("Training models...")
    results = predictor.train_models(targets)
    
    # Generate report
    predictor.generate_report(results)
    
    # Make predictions
    print("\nMaking predictions with latest data...")
    predictions = predictor.predict()
    print("Predictions:")
    for target, pred in predictions.items():
        print(f"  {target}: {pred:.2f}")
    
    # Plot results
    for target_name in targets.keys():
        predictor.plot_predictions_vs_actual(results, target_name)
        if target_name in predictor.feature_importance:
            predictor.plot_feature_importance(target_name)
    
    # Save models
    predictor.save_models('well_performance_models.pkl')
    
    return predictor, results


if __name__ == "__main__":
    predictor, results = main()