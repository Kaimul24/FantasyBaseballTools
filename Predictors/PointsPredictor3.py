import sys
import os
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
training_dir = "training_results"  # or any other directory name you prefer
models = "models"  # or any other directory name you prefer
os.makedirs(models, exist_ok=True)
os.makedirs(training_dir, exist_ok=True)

from DataProcessing.DataProcessing import WeightedDatasetSplit
from DataProcessing.BatterDataProcessing import BatterDataProcessing
from DataProcessing.StarterDataProcessing import StarterDataProcessing
from DataProcessing.RelieverDataProcessing import RelieverDataProcessing
from DataProcessing.DataProcessingComposition import TrainingDataProcessor
import pandas as pd
import numpy as np
from typing import List
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from abc import ABC, abstractmethod

class BaseModel(ABC):

    @staticmethod
    def train_model(windows: List[WeightedDatasetSplit]) -> xgb.XGBRegressor:
        '''Train single XGBoost model and predict on validation data'''

        # Create single model instance
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            tree_method='hist',
            n_estimators=430,
            learning_rate=0.065,
            max_depth=2,
            min_child_weight=6,
            subsample=0.75,
            colsample_bytree=0.775,
            gamma=0.0175,
            reg_alpha=0.075,
            reg_lambda=0.52,
            random_state=42,
        )
        
        # Accumulate training data across windows
        all_X = []
        all_y = []
        
        # Collect all training data
        for window in windows:
            weighted_data = window['weighted_data']
            
            feature_cols = [col for col in weighted_data.columns 
                        if col != 'PlayerName' and col != 'TotalPoints']

            if 'TotalPoints' not in weighted_data.columns or not feature_cols:
                print(f"Skipping window with training years {window['train_years']}: missing required columns.")
                continue
                
            train_df = weighted_data[['PlayerName'] + feature_cols + ['TotalPoints']].dropna()
            
            X = train_df[feature_cols]
            y = train_df['TotalPoints']
            
            all_X.append(X)
            all_y.append(y)
        
        # Train model on combined data
        X_full = pd.concat(all_X)
        y_full = pd.concat(all_y)
        
        model.fit(X_full, y_full, verbose=False)
        
        return model
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Calculate RMSE, MAE, and R² score for predictions
        
        Args:
            y_true: Array of actual values
            y_pred: Array of predicted values
        
        Returns:
            Dictionary containing RMSE, MAE, and R² score
        """
        # Ensure inputs are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Check for valid input
        if len(y_true) != len(y_pred):
            raise ValueError("Length of actual and predicted values must match")
        if len(y_true) == 0:
            raise ValueError("Empty arrays provided")
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

    @staticmethod
    def save_trained_model(model: xgb.XGBRegressor, filepath: str) -> None:
        """
        Save a trained XGBoost model to disk
        
        Args:
            model: Trained XGBRegressor model
            filepath: Path where the model will be saved
        """
        model.save_model(filepath)
        print(f"Model successfully saved to {filepath}")
        
    @staticmethod
    @abstractmethod
    def predict_model(model: xgb.XGBRegressor, windows: List[WeightedDatasetSplit]) -> List[WeightedDatasetSplit]:
        pass

class BatterModel(BaseModel):
    """Concrete implementation for batter modeling"""
    
    @staticmethod
    def predict_model(model: xgb.XGBRegressor, windows: List[WeightedDatasetSplit]) -> List[WeightedDatasetSplit]:
        """
        Make predictions on test data for each window using the trained model
        
        Args:
            model: Trained XGBRegressor model
            windows: List of data windows containing test data
            
        Returns:
            List of windows with predictions added
        """
        # Make predictions on test data for each window
        invalid_cols = ['1B', '2B', '3B', 'HR', 'R', 'RBI', 'HBP', 'SB']
        for window in windows:
            test_data = window['test_data']
            test_year = window['test_year']
            
            # Get validation features (removing year prefix)
            val_features = [col for col in test_data.columns 
                          if col != 'PlayerName' and col != 'TotalPoints' 
                          and col not in invalid_cols]
            
            if not val_features:
                print(f"Skipping predictions for validation year {test_year}: no features found")
                continue
                
            X_val = test_data[val_features]
            
            # Check if TotalPoints exists in test data
            if 'TotalPoints' in test_data.columns:
                y_val = test_data['TotalPoints']
            else:
                y_val = pd.Series([np.nan] * len(test_data))
                print(f"Warning: No TotalPoints found in test data for year {test_year}")
            
            val_predictions = model.predict(X_val)
            
            window['model'] = model
            window['predictions'] = pd.DataFrame({
                'PlayerName': test_data['PlayerName'],
                'predicted_TotalPoints': val_predictions,
                'actual_TotalPoints': y_val.values
            })
        
        return windows

class StarterModel(BaseModel):
    """Concrete implementation for starting pitcher modeling"""
    
    @staticmethod
    def predict_model(model: xgb.XGBRegressor, windows: List[WeightedDatasetSplit]) -> List[WeightedDatasetSplit]:
        """
        Make predictions on test data for each window using the trained model
        
        Args:
            model: Trained XGBRegressor model
            windows: List of data windows containing test data
            
        Returns:
            List of windows with predictions added
        """
        # Make predictions on test data for each window
        invalid_cols = ['W', 'SO', 'ER', 'BB', 'H', 'HBP', 'IP']
        for window in windows:
            test_data = window['test_data']
            test_year = window['test_year']
            
            # Get validation features (removing year prefix)
            val_features = [col for col in test_data.columns 
                          if col != 'PlayerName' and col != 'TotalPoints' 
                          and col not in invalid_cols]
            
            if not val_features:
                print(f"Skipping predictions for validation year {test_year}: no features found")
                continue
                
            X_val = test_data[val_features]
            
            # Check if TotalPoints exists in test data
            if 'TotalPoints' in test_data.columns:
                y_val = test_data['TotalPoints']
            else:
                y_val = pd.Series([np.nan] * len(test_data))
                print(f"Warning: No TotalPoints found in test data for year {test_year}")
            
            val_predictions = model.predict(X_val)
            
            window['model'] = model
            window['predictions'] = pd.DataFrame({
                'PlayerName': test_data['PlayerName'],
                'predicted_TotalPoints': val_predictions,
                'actual_TotalPoints': y_val.values
            })
        
        return windows
    
class RelieverModel(BaseModel):
    """Concrete implementation for starting pitcher modeling"""
    
    @staticmethod
    def predict_model(model: xgb.XGBRegressor, windows: List[WeightedDatasetSplit]) -> List[WeightedDatasetSplit]:
        """
        Make predictions on test data for each window using the trained model
        
        Args:
            model: Trained XGBRegressor model
            windows: List of data windows containing test data
            
        Returns:
            List of windows with predictions added
        """
        # Make predictions on test data for each window
        invalid_cols = ['SV', 'HLD', 'SO', 'ER', 'BB', 'H', 'HBP', 'IP']
        for window in windows:
            test_data = window['test_data']
            test_year = window['test_year']
            
            # Get validation features (removing year prefix)
            val_features = [col for col in test_data.columns 
                          if col != 'PlayerName' and col != 'TotalPoints' 
                          and col not in invalid_cols]
            
            if not val_features:
                print(f"Skipping predictions for validation year {test_year}: no features found")
                continue
                
            X_val = test_data[val_features]
            
            # Check if TotalPoints exists in test data
            if 'TotalPoints' in test_data.columns:
                y_val = test_data['TotalPoints']
            else:
                y_val = pd.Series([np.nan] * len(test_data))
                print(f"Warning: No TotalPoints found in test data for year {test_year}")
            
            val_predictions = model.predict(X_val)
            
            window['model'] = model
            window['predictions'] = pd.DataFrame({
                'PlayerName': test_data['PlayerName'],
                'predicted_TotalPoints': val_predictions,
                'actual_TotalPoints': y_val.values
            })
        
        return windows

def main():
    ##### BATTERS #####
    print("=" * 50 + " BATTERS " + "=" * 50)
    print("Loading data...\n")
    data_processor = BatterDataProcessing()
    data_processor.filter_data()
    data_processor.calc_fantasy_points()
    
    # Use the data processor to prepare windows
    data_processor = TrainingDataProcessor(data_processor)
    processed_windows = data_processor.prepare_data()

    # Print all data to file
    with open(os.path.join(training_dir,'batter_data.txt'), 'w') as f:
        for i, window in enumerate(processed_windows):
            f.write(f"\n{'='*80}\n")
            f.write(f"Window {i+1}:\n")
            f.write(f"Training years: {window['train_years']}\n")
            f.write(f"Test year: {window['test_year']}\n")
            
            f.write("\nWeighted Training Data:\n")
            f.write(f"Shape: {window['weighted_data'].shape}\n")
            f.write(window['weighted_data'].to_string())
            
            f.write("\n\nTest Data:\n")
            f.write(f"Shape: {window['test_data'].shape}\n")
            f.write(window['test_data'].to_string())
            f.write(f"\n{'='*80}\n")

    # Train model
    print("Training models...\n")
    model = BatterModel.train_model(processed_windows)

    # Save trained model
    print("Saving model...\n")
    BatterModel.save_trained_model(model, os.path.join(models,"batter_model.json"))

    # Make predictions for each window
    print("Making predictions...\n")
    tested_windows =  BatterModel.predict_model(model, processed_windows)

    with open(os.path.join(training_dir,"batter_predictions.txt"), "w") as f:
        # Track overall metrics
        all_metrics = []
        
        for i, window in enumerate(tested_windows):
            if "predictions" not in window:
                f.write(f"Window {i+1}: No predictions available.\n")
                continue
            
            f.write(f"Window {i+1}:\n")
            f.write(f"Training years: {window['train_years']}, Test year: {window['test_year']}\n")
            predictions = window["predictions"].copy()
            
            # Calculate metrics
            metrics =  RelieverModel.calculate_metrics(
                predictions['actual_TotalPoints'],
                predictions['predicted_TotalPoints']
            )
            all_metrics.append(metrics)
            
            # Write metrics
            f.write("\nTest Metrics:\n")
            f.write(f"RMSE: {metrics['rmse']:.2f}\n")
            f.write(f"MAE: {metrics['mae']:.2f}\n")
            f.write(f"R² Score: {metrics['r2']:.3f}\n\n")
            
            # Compute and write predictions
            predictions["percent_diff"] = predictions.apply(
                lambda row: abs(row["predicted_TotalPoints"] - row["actual_TotalPoints"]) / row["actual_TotalPoints"] * 100
                            if row["actual_TotalPoints"] != 0 else 0, axis=1)
            
            f.write(f"{'PlayerName':<20}{'Predicted':>15}{'Actual':>15}{'Percent Diff (%)':>20}\n")
            f.write("-" * 70 + "\n")
            
            for _, row in predictions.iterrows():
                f.write(f"{row['PlayerName']:<20}{row['predicted_TotalPoints']:>15.2f}"
                        f"{row['actual_TotalPoints']:>15.2f}{row['percent_diff']:>20.2f}\n")
            f.write("\n" + "=" * 80 + "\n\n")
        
        # Write average metrics across all windows
        if all_metrics:
            avg_rmse = np.mean([m['rmse'] for m in all_metrics])
            avg_mae = np.mean([m['mae'] for m in all_metrics])
            avg_r2 = np.mean([m['r2'] for m in all_metrics])
            f.write("\nOverall Model Performance:\n")
            f.write(f"Average RMSE: {avg_rmse:.2f}\n")
            f.write(f"Average MAE: {avg_mae:.2f}\n")
            f.write(f"Average R² Score: {avg_r2:.3f}\n")
            f.write("=" * 80 + "\n")
            print(f"Overall Model Performance: \n\tAverage RMSE: {avg_rmse:.2f} \n\tAverage MAE: {avg_mae:.2f} \n\tAverage R² Score: {avg_r2:.3f}")

    ##### STARTERS #####
    print("=" * 50 + " STARTERS " + "=" * 50)
    print("Loading data...\n")
    data_processor = StarterDataProcessing()
    data_processor.filter_data()
    data_processor.calc_fantasy_points()
    
    # Use the data processor to prepare windows
    data_processor = TrainingDataProcessor(data_processor)
    processed_windows = data_processor.prepare_data()

    # Print all data to file
    with open(os.path.join(training_dir,'starter_data.txt'), 'w') as f:
        for i, window in enumerate(processed_windows):
            f.write(f"\n{'='*80}\n")
            f.write(f"Window {i+1}:\n")
            f.write(f"Training years: {window['train_years']}\n")
            f.write(f"Test year: {window['test_year']}\n")
            
            f.write("\nWeighted Training Data:\n")
            f.write(f"Shape: {window['weighted_data'].shape}\n")
            f.write(window['weighted_data'].to_string())
            
            f.write("\n\nTest Data:\n")
            f.write(f"Shape: {window['test_data'].shape}\n")
            f.write(window['test_data'].to_string())
            f.write(f"\n{'='*80}\n")

    # Train model
    print("Training models...\n")
    model = StarterModel.train_model(processed_windows)

    # Save trained model
    print("Saving model...\n")
    StarterModel.save_trained_model(model, os.path.join(models,"starter_model.json"))

    # Make predictions for each window
    print("Making predictions...\n")
    tested_windows =  StarterModel.predict_model(model, processed_windows)

    with open(os.path.join(training_dir,"starter_predictions.txt"), "w") as f:
        # Track overall metrics
        all_metrics = []
        
        for i, window in enumerate(tested_windows):
            if "predictions" not in window:
                f.write(f"Window {i+1}: No predictions available.\n")
                continue
            
            f.write(f"Window {i+1}:\n")
            f.write(f"Training years: {window['train_years']}, Test year: {window['test_year']}\n")
            predictions = window["predictions"].copy()
            
            # Calculate metrics
            metrics =  RelieverModel.calculate_metrics(
                predictions['actual_TotalPoints'],
                predictions['predicted_TotalPoints']
            )
            all_metrics.append(metrics)
            
            # Write metrics
            f.write("\nTest Metrics:\n")
            f.write(f"RMSE: {metrics['rmse']:.2f}\n")
            f.write(f"MAE: {metrics['mae']:.2f}\n")
            f.write(f"R² Score: {metrics['r2']:.3f}\n\n")
            
            # Compute and write predictions
            predictions["percent_diff"] = predictions.apply(
                lambda row: abs(row["predicted_TotalPoints"] - row["actual_TotalPoints"]) / row["actual_TotalPoints"] * 100
                            if row["actual_TotalPoints"] != 0 else 0, axis=1)
            
            f.write(f"{'PlayerName':<20}{'Predicted':>15}{'Actual':>15}{'Percent Diff (%)':>20}\n")
            f.write("-" * 70 + "\n")
            
            for _, row in predictions.iterrows():
                f.write(f"{row['PlayerName']:<20}{row['predicted_TotalPoints']:>15.2f}"
                        f"{row['actual_TotalPoints']:>15.2f}{row['percent_diff']:>20.2f}\n")
            f.write("\n" + "=" * 80 + "\n\n")
        
        # Write average metrics across all windows
        if all_metrics:
            avg_rmse = np.mean([m['rmse'] for m in all_metrics])
            avg_mae = np.mean([m['mae'] for m in all_metrics])
            avg_r2 = np.mean([m['r2'] for m in all_metrics])
            f.write("\nOverall Model Performance:\n")
            f.write(f"Average RMSE: {avg_rmse:.2f}\n")
            f.write(f"Average MAE: {avg_mae:.2f}\n")
            f.write(f"Average R² Score: {avg_r2:.3f}\n")
            f.write("=" * 80 + "\n")
            print(f"Overall Model Performance: \n\tAverage RMSE: {avg_rmse:.2f} \n\tAverage MAE: {avg_mae:.2f} \n\tAverage R² Score: {avg_r2:.3f}")

    ##### RELIEVERS #####
    print("=" * 50 + " RELIEVERS " + "=" * 50)
    print("Loading data...\n")
    data_processor = RelieverDataProcessing()
    data_processor.filter_data()
    data_processor.calc_fantasy_points()
    
    # Use the data processor to prepare windows
    data_processor = TrainingDataProcessor(data_processor)
    processed_windows = data_processor.prepare_data()

    # Print all data to file
    with open(os.path.join(training_dir,'reliever_data.txt'), 'w') as f:
        for i, window in enumerate(processed_windows):
            f.write(f"\n{'='*80}\n")
            f.write(f"Window {i+1}:\n")
            f.write(f"Training years: {window['train_years']}\n")
            f.write(f"Test year: {window['test_year']}\n")
            
            f.write("\nWeighted Training Data:\n")
            f.write(f"Shape: {window['weighted_data'].shape}\n")
            f.write(window['weighted_data'].to_string())
            
            f.write("\n\nTest Data:\n")
            f.write(f"Shape: {window['test_data'].shape}\n")
            f.write(window['test_data'].to_string())
            f.write(f"\n{'='*80}\n")

    # Train model
    print("Training models...\n")
    model = RelieverModel.train_model(processed_windows)

    # Save trained model
    print("Saving model...\n")
    RelieverModel.save_trained_model(model, os.path.join(models,"reliever_model.json"))

    # Make predictions for each window
    print("Making predictions...\n")
    tested_windows =  RelieverModel.predict_model(model, processed_windows)

    with open(os.path.join(training_dir,"reliever_predictions.txt"), "w") as f:
        # Track overall metrics
        all_metrics = []
        
        for i, window in enumerate(tested_windows):
            if "predictions" not in window:
                f.write(f"Window {i+1}: No predictions available.\n")
                continue
            
            f.write(f"Window {i+1}:\n")
            f.write(f"Training years: {window['train_years']}, Test year: {window['test_year']}\n")
            predictions = window["predictions"].copy()
            
            # Calculate metrics
            metrics =  RelieverModel.calculate_metrics(
                predictions['actual_TotalPoints'],
                predictions['predicted_TotalPoints']
            )
            all_metrics.append(metrics)
            
            # Write metrics
            f.write("\nTest Metrics:\n")
            f.write(f"RMSE: {metrics['rmse']:.2f}\n")
            f.write(f"MAE: {metrics['mae']:.2f}\n")
            f.write(f"R² Score: {metrics['r2']:.3f}\n\n")
            
            # Compute and write predictions
            predictions["percent_diff"] = predictions.apply(
                lambda row: abs(row["predicted_TotalPoints"] - row["actual_TotalPoints"]) / row["actual_TotalPoints"] * 100
                            if row["actual_TotalPoints"] != 0 else 0, axis=1)
            
            f.write(f"{'PlayerName':<20}{'Predicted':>15}{'Actual':>15}{'Percent Diff (%)':>20}\n")
            f.write("-" * 70 + "\n")
            
            for _, row in predictions.iterrows():
                f.write(f"{row['PlayerName']:<20}{row['predicted_TotalPoints']:>15.2f}"
                        f"{row['actual_TotalPoints']:>15.2f}{row['percent_diff']:>20.2f}\n")
            f.write("\n" + "=" * 80 + "\n\n")
        
        # Write average metrics across all windows
        if all_metrics:
            avg_rmse = np.mean([m['rmse'] for m in all_metrics])
            avg_mae = np.mean([m['mae'] for m in all_metrics])
            avg_r2 = np.mean([m['r2'] for m in all_metrics])
            f.write("\nOverall Model Performance:\n")
            f.write(f"Average RMSE: {avg_rmse:.2f}\n")
            f.write(f"Average MAE: {avg_mae:.2f}\n")
            f.write(f"Average R² Score: {avg_r2:.3f}\n")
            f.write("=" * 80 + "\n")
            print(f"Overall Model Performance: \n\tAverage RMSE: {avg_rmse:.2f} \n\tAverage MAE: {avg_mae:.2f} \n\tAverage R² Score: {avg_r2:.3f}")


if __name__ == "__main__":
    main()