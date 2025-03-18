import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
training_dir = "training_results"
models = "models" 
os.makedirs(models, exist_ok=True)
os.makedirs(training_dir, exist_ok=True)

from DataProcessing.DataProcessing import WeightedDatasetSplit
from DataProcessing.BatterDataProcessing import BatterDataProcessing
from DataProcessing.StarterDataProcessing import StarterDataProcessing
from DataProcessing.RelieverDataProcessing import RelieverDataProcessing
from DataProcessing.DataPipelines import TrainingDataPrep, PredictionDataPrep, LeagueType
from FangraphsScraper.fangraphsScraper import PositionCategory  
from enum import Enum
import pandas as pd
import numpy as np
from typing import List, Union
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

STARTER_PARAMS = {
                'n_estimators': [430],
                'learning_rate': [0.06],
                'max_depth': [2],
                'min_child_weight': [6],
                'subsample': [0.75],
                'colsample_bytree': [0.76],
                'gamma': [0],
                'reg_alpha': [0.07],
                'reg_lambda': [0.52],
            }
BATTER_PARAMS = {
                'n_estimators': [430],
                'learning_rate': [0.065],
                'max_depth': [2],
                'min_child_weight': [6],
                'subsample': [0.75],
                'colsample_bytree': [0.775],
                'gamma': [0.0175],
                'reg_alpha': [0.075],
                'reg_lambda': [0.52],
            }   

# TODO
RELIEVER_PARAMS = {
                'n_estimators': [430],
                'learning_rate': [0.06],
                'max_depth': [2],
                'min_child_weight': [6],
                'subsample': [0.75],
                'colsample_bytree': [0.76],
                'gamma': [0],
                'reg_alpha': [0.07],
                'reg_lambda': [0.52],
            }

class Model():
    def __init__(self, data_prep: Union[TrainingDataPrep, PredictionDataPrep], league_type: LeagueType = LeagueType.POINTS, category: str = "TotalPoints"):
        self.league_type = league_type
        self.position_category = data_prep.data_processor.position_category

        if (self.league_type != data_prep.league_type):
            raise ValueError("Mismatched league types between model and data processor")

        if self.league_type == LeagueType.POINTS:
            self.category = "TotalPoints"
            self.invalid_cols = data_prep.data_processor.get_counting_stats()
        else:
            self.category = category
            self.invalid_cols = []

    def hyperparam_tune(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
            """Perform hyperparameter tuning using GridSearchCV"""
            print("Tuning Hyperparameters...\n")
            tscv = TimeSeriesSplit(n_splits=5)
            
            param_grid = {
                'n_estimators': [430],
                'learning_rate': [0.06],
                'max_depth': [2],
                'min_child_weight': [6],
                'subsample': [0.75],
                'colsample_bytree': [0.76],
                'gamma': [0],
                'reg_alpha': [0.07],
                'reg_lambda': [0.52],
            }

            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                tree_method='hist',
                random_state=42
            )

            grid_search = GridSearchCV(
                model, param_grid, cv=tscv, scoring='r2', 
                n_jobs=-1, verbose=1
            )

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            with open(os.path.join(training_dir, "best_hyperparams.txt"), "a") as f:
                f.write(f"Best hyperparameters: {grid_search.best_params_}\n")
                f.write(f"Best R² Score: {grid_search.best_score_}\n\n")
            print(f"Best hyperparameters: {grid_search.best_params_}\n")

            return best_model

    def train_model(self, windows: List[WeightedDatasetSplit], tune_hyperparams: bool = False) -> xgb.XGBRegressor:
        '''Train single XGBoost model on combined data from all windows'''
        
        # Determine which parameter set to use based on model type
        if self.position_category == PositionCategory.BATTER:
            params = BATTER_PARAMS
        elif self.position_category == PositionCategory.SP:
            params = STARTER_PARAMS 
        elif self.position_category == PositionCategory.RP:
            params = RELIEVER_PARAMS
        else:
            raise TypeError("Unknown model type")
        
        # Accumulate training data across windows
        all_X = []
        all_y = []
        
        # Collect all training data
        for window in windows:
            weighted_data = window['weighted_data']
            
            feature_cols = [col for col in weighted_data.columns 
                        if col != 'PlayerName' and col != self.category]

            if self.category not in weighted_data.columns or not feature_cols:
                print(f"Skipping window with training years {window['train_years']}: missing required column: {self.category}.")
                continue
                
            train_df = weighted_data[['PlayerName'] + feature_cols + [self.category]].dropna()

            X = train_df[feature_cols]

            if self.league_type == LeagueType.CATEGORIES:
                if self.category == 'TB':
                    X = X.drop(['1B', '2B', '3B', 'HR'], axis=1)
                elif self.category == 'NSB':
                    X = X.drop(['SB', 'CS'], axis=1)
                elif self.category == 'OPS':
                    X = X.drop(['OBP', 'SLG'], axis=1)

            y = train_df[self.category]
            
            all_X.append(X)
            all_y.append(y)
        
        # Train model on combined data
        X_full = pd.concat(all_X)
        y_full = pd.concat(all_y)

        if tune_hyperparams:
            model = self.hyperparam_tune(X_full, y_full)
        else:
            # Create model with parameter values from the first item of each parameter list
            model_params = {k: v[0] for k, v in params.items()}
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                tree_method='hist',
                random_state=42,
                **model_params
            )
        
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
     
    def predict_model(self, model: xgb.XGBRegressor, windows: List[WeightedDatasetSplit]) -> List[WeightedDatasetSplit]:
        """
        Make predictions on test data for each window using the trained model.
        Handles both points and category-based league types.
        
        Args:
            model: Trained XGBRegressor model
            windows: List of data windows containing test data
            target_col: Target column to predict (default: "TotalPoints")
            league_type: Type of league (points or categories)
            
        Returns:
            List of windows with predictions added
        """
        # Make predictions on test data for each window
        
        for window in windows:
            test_data = window['test_data']
            test_year = window['test_year']

            # Get validation features (removing year prefix and invalid columns)
            val_features = [col for col in test_data.columns 
                          if col != 'PlayerName' and col != self.category 
                          and col not in self.invalid_cols]
            
            if not val_features:
                print(f"Skipping predictions for validation year {test_year}: no features found")
                continue
                
            X_val = test_data[val_features]
            if self.league_type == LeagueType.CATEGORIES:
                if self.category == 'TB':
                    X_val = X_val.drop(['1B', '2B', '3B', 'HR'], axis=1)
                elif self.category == 'NSB':
                    X_val = X_val.drop(['SB', 'CS'], axis=1)
                elif self.category == 'OPS':
                    X_val = X_val.drop(['OBP', 'SLG'], axis=1)
            
            # Check if target column exists in test data
            if self.category in test_data.columns:
                y_val = test_data[self.category]
            else:
                y_val = pd.Series([np.nan] * len(test_data))
                print(f"Warning: No {self.category} found in test data for year {test_year}")
            
            val_predictions = model.predict(X_val)
            
            window['model'] = model
            window['predictions'] = pd.DataFrame({
                'PlayerName': test_data['PlayerName'],
                f'predicted_{self.category}': val_predictions,
                f'actual_{self.category}': y_val.values
            })
        
        return windows

def main():
    ##### BATTERS #####
    print("=" * 50 + " BATTERS " + "=" * 50)
    print("Loading data...\n")
    batter_data = BatterDataProcessing()
    batter_data.filter_and_calc_points()
    
    # Use the data processor to prepare windows
    data_prep = TrainingDataPrep(batter_data)
    processed_windows = data_prep.prepare_data()

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
    batter_model = Model(data_prep)
    model = batter_model.train_model(processed_windows, tune_hyperparams=False)

    # Save trained model
    print("Saving model...\n")
    Model.save_trained_model(model, os.path.join(models,"batter_model.json"))

    # Make predictions for each window
    print("Making predictions...\n")
    tested_windows = batter_model.predict_model(model, processed_windows)

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
            metrics =  Model.calculate_metrics(
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
    starter_data = StarterDataProcessing()
    starter_data.filter_and_calc_points()
    
    # Use the data processor to prepare windows
    data_processor = TrainingDataPrep(starter_data)
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
    starter_model = Model(data_processor)
    model = starter_model.train_model(processed_windows, tune_hyperparams=False)

    # Save trained model
    print("Saving model...\n")
    Model.save_trained_model(model, os.path.join(models,"starter_model.json"))

    # Make predictions for each window
    print("Making predictions...\n")
    tested_windows = starter_model.predict_model(model, processed_windows)

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
            metrics = Model.calculate_metrics(
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
    reliever_data = RelieverDataProcessing()
    reliever_data.filter_and_calc_points()
    
    # Use the data processor to prepare windows
    data_processor = TrainingDataPrep(reliever_data)
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
    reliever_model = Model(data_processor)
    model = reliever_model.train_model(processed_windows, tune_hyperparams=False)

    # Save trained model
    print("Saving model...\n")
    Model.save_trained_model(model, os.path.join(models,"reliever_model.json"))

    # Make predictions for each window
    print("Making predictions...\n")
    tested_windows = reliever_model.predict_model(model, processed_windows)

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
            metrics =  Model.calculate_metrics(
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