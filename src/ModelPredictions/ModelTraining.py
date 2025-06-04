from ..DataProcessing.DataProcessing import WeightedDatasetSplit
from ..DataProcessing.BatterDataProcessing import BatterDataProcessing
from ..DataProcessing.StarterDataProcessing import StarterDataProcessing
from ..DataProcessing.RelieverDataProcessing import RelieverDataProcessing
from ..DataProcessing.DataPipelines import TrainingDataPrep, PredictionDataPrep, LeagueType
from ..FangraphsScraper.fangraphsScraper import PositionCategory  
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Any, Tuple, cast
from numpy.typing import ArrayLike
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from pathlib import Path

from config import TRAINING_DIR, MODELS_DIR

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

class Model:
    def __init__(self, data_prep: TrainingDataPrep, league_type: LeagueType = LeagueType.POINTS, category: str = "TotalPoints"):
        """
        Initialize Model with training data preparation, league type, and prediction category.
        
        Args:
            data_prep: Data preparation object with processed data
            league_type: Type of league (POINTS or CATEGORIES)
            category: Target category to predict (default: "TotalPoints")
        """
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

    def _hyperparam_tune(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training target values
            
        Returns:
            Tuned XGBoost regressor with best hyperparameters
        """
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
        with open((TRAINING_DIR / "best_hyperparams.txt"), "a") as f:
            f.write(f"Best hyperparameters: {grid_search.best_params_}\n")
            f.write(f"Best R² Score: {grid_search.best_score_}\n\n")
        print(f"Best hyperparameters: {grid_search.best_params_}\n")

        return best_model

        
    def train_model(self, windows: List[WeightedDatasetSplit], tune_hyperparams: bool = False) -> xgb.XGBRegressor:
        """
        Train single XGBoost model on combined data from all windows.
        
        Args:
            windows: List of data windows containing training data
            tune_hyperparams: Whether to perform hyperparameter tuning
            
        Returns:
            Trained XGBoost regressor
        """
        if self.position_category == PositionCategory.BATTER:
            params = BATTER_PARAMS
        elif self.position_category == PositionCategory.SP:
            params = STARTER_PARAMS 
        elif self.position_category == PositionCategory.RP:
            params = RELIEVER_PARAMS
        else:
            raise TypeError("Unknown model type")
        
        all_X = []
        all_y = []
        
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
                if self.position_category == PositionCategory.BATTER:
                    if self.category == 'TB':
                        X = X.drop(['1B', '2B', '3B', 'HR'], axis=1)
                    elif self.category == 'NSB':
                        X = X.drop(['SB', 'CS'], axis=1)
                    elif self.category == 'OPS':
                        X = X.drop(['OBP', 'SLG'], axis=1)
                else:
                    if self.category == 'SO':
                        X = X.drop(['SO'], axis=1, errors='ignore')
                    elif self.category == 'WHIP':
                        X = X.drop(['WHIP'], axis=1, errors='ignore')
                    elif self.category == 'ERA':
                        X = X.drop(['ERA'], axis=1, errors='ignore')
                    elif self.category == 'QS':
                        X = X.drop(['QS'], axis=1, errors='ignore')
                    elif self.category == 'NSVH':
                        X = X.drop(['SV', 'HLD', 'BS'], axis=1)

            y = train_df[self.category]
            
            all_X.append(X)
            all_y.append(y)
        
        X_full = pd.concat(all_X)
        y_full = pd.concat(all_y)

        if tune_hyperparams:
            model = self._hyperparam_tune(X_full, y_full)
        else:
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
    def calculate_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
        """
        Calculate RMSE, MAE, and R² score for predictions.
        
        Args:
            y_true: Array of actual values
            y_pred: Array of predicted values
        
        Returns:
            Dictionary containing RMSE, MAE, and R² score
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if len(y_true) != len(y_pred):
            raise ValueError("Length of actual and predicted values must match")
        if len(y_true) == 0:
            raise ValueError("Empty arrays provided")
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
    @staticmethod
    def save_trained_model(model: xgb.XGBRegressor, filepath: Path) -> None:
        """
        Save a trained XGBoost model to disk.
        
        Args:
            model: Trained XGBRegressor model
            filepath: Path where the model will be saved
        """
        model.save_model(str(filepath))
        print(f"Model successfully saved to {filepath}")
        
    def predict_model(self, model: xgb.XGBRegressor, windows: List[WeightedDatasetSplit]) -> List[WeightedDatasetSplit]:
        """
        Make predictions on test data for each window using the trained model.
        
        Args:
            model: Trained XGBRegressor model
            windows: List of data windows containing test data
            
        Returns:
            List of windows with predictions added
        """
        for window in windows:
            test_data = window['test_data']
            test_year = window['test_year']

            val_features = [col for col in test_data.columns 
                          if col != 'PlayerName' and col != self.category 
                          and col not in self.invalid_cols]
            
            if not val_features:
                print(f"Skipping predictions for validation year {test_year}: no features found")
                continue
                
            X_val = test_data[val_features]

            if self.league_type == LeagueType.CATEGORIES:
                if self.position_category == PositionCategory.BATTER:
                    if self.category == 'TB':
                        X_val = X_val.drop(['1B', '2B', '3B', 'HR'], axis=1)
                    elif self.category == 'NSB':
                        X_val = X_val.drop(['SB', 'CS'], axis=1)
                    elif self.category == 'OPS':
                        X_val = X_val.drop(['OBP', 'SLG'], axis=1)
                else:
                    if self.category == 'SO':
                        X_val = X_val.drop(['SO'], axis=1, errors='ignore')
                    elif self.category == 'WHIP':
                        X_val = X_val.drop(['WHIP'], axis=1, errors='ignore')
                    elif self.category == 'ERA':
                        X_val = X_val.drop(['ERA'], axis=1, errors='ignore')
                    elif self.category == 'QS':
                        X_val = X_val.drop(['QS'], axis=1, errors='ignore')
                    elif self.category == 'NSVH':
                        X_val = X_val.drop(['SV', 'HLD', 'BS'], axis=1)
            
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
    
def train_and_evaluate_model(position_type: PositionCategory, league_type: LeagueType, 
                           start_year: int = 2019, end_year: int = 2024) -> None:
    """
    Train and evaluate models for the specified position type and league type.
    
    Args:
        position_type: PositionCategory to train models for
        league_type: LeagueType (POINTS or CATEGORIES)
        start_year: First year of data to use
        end_year: Last year of data to use
    """
    position_map = {
        PositionCategory.BATTER: BatterDataProcessing,
        PositionCategory.SP: StarterDataProcessing,
        PositionCategory.RP: RelieverDataProcessing
    }
    
    if position_type not in position_map:
        raise ValueError(f"Invalid position type: {position_type}. Must be one of {list(position_map.keys())}")
    
    position_name = str(position_type)[17:].lower()
    print("=" * 50 + f" {str(position_type)[17:]} ({str(league_type)[11:]}) " + "=" * 50)
    print("Loading data...\n")
    
    data_processor = position_map[position_type](league_type=league_type, start_year=start_year, end_year=end_year)
    
    if league_type == LeagueType.POINTS:
        data_processor.filter_and_calc_points()
    else:
        data_processor.filter_data()
    
    data_prep = TrainingDataPrep(data_processor)
    processed_windows = cast(List[WeightedDatasetSplit], data_prep.prepare_data())
    
    with open((TRAINING_DIR / f'{position_type}_{league_type.name.lower()}_data.txt'), 'w') as f:
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

    if league_type == LeagueType.POINTS:
        print("Training model for TotalPoints...\n")
        model_instance = Model(data_prep)
        trained_model = model_instance.train_model(processed_windows, tune_hyperparams=False)
        
        model_filepath = MODELS_DIR / f"{str(position_type)[17:]}_model.json"
        Model.save_trained_model(trained_model, model_filepath)
        
        tested_windows = model_instance.predict_model(trained_model, processed_windows)
        
        output_file = TRAINING_DIR / f"{position_type}_predictions.txt"
        _evaluate_and_write_predictions(tested_windows, output_file, "TotalPoints")
    
    else:
        for category in data_processor.stat_categories:
            print(f"Training model for {category}...")
            model_instance = Model(data_prep, league_type=LeagueType.CATEGORIES, category=category)
            trained_model = model_instance.train_model(processed_windows)
            
            model_filepath = MODELS_DIR / f"{position_name}_{category}_model.json"
            Model.save_trained_model(trained_model, model_filepath)
            
            tested_windows = model_instance.predict_model(trained_model, processed_windows)
            
            output_file = TRAINING_DIR / f"category_{position_name}_{category}_predictions.txt"
            _evaluate_and_write_predictions(tested_windows, output_file, category)

def _evaluate_and_write_predictions(tested_windows: List[WeightedDatasetSplit], 
                                  output_file: Path, 
                                  category: str) -> None:
    """
    Evaluate model predictions and write results to a file.
    
    Args:
        tested_windows: List of windows with predictions
        output_file: Path to output file
        category: Category name (TotalPoints or specific stat category)
    """
    with open(output_file, "w") as f:
        all_metrics = []
        
        for i, window in enumerate(tested_windows):
            preds = window.get("predictions")
            if preds is None:
                f.write(f"Window {i+1}: No predictions available.\n")
                continue
            
            if category != "TotalPoints":
                f.write(f"Stat: {category}\n")
                
            f.write(f"Window {i+1}:\n")
            f.write(f"Training years: {window['train_years']}, Test year: {window['test_year']}\n")
            predictions = preds.copy()
            
            metrics = Model.calculate_metrics(
                predictions[f'actual_{category}'].values,
                predictions[f'predicted_{category}'].values
            )
            all_metrics.append(metrics)
            
            f.write("\nTest Metrics:\n")
            f.write(f"RMSE: {metrics['rmse']:.2f}\n")
            f.write(f"MAE: {metrics['mae']:.2f}\n")
            f.write(f"R² Score: {metrics['r2']:.3f}\n\n")
            
            predictions["percent_diff"] = predictions.apply(
                lambda row: abs(row[f'predicted_{category}'] - row[f'actual_{category}']) / row[f'actual_{category}'] * 100
                            if row[f'actual_{category}'] != 0 else 0, axis=1)
            
            f.write(f"{'PlayerName':<20}{'Predicted':>15}{'Actual':>15}{'Percent Diff (%)':>20}\n")
            f.write("-" * 70 + "\n")
            
            for _, row in predictions.iterrows():
                f.write(f"{row['PlayerName']:<20}{row[f'predicted_{category}']:>15.2f}"
                        f"{row[f'actual_{category}']:>15.2f}{row['percent_diff']:>20.2f}\n")
            f.write("\n" + "=" * 80 + "\n\n")
        
        if all_metrics:
            avg_rmse = np.mean([m['rmse'] for m in all_metrics])
            avg_mae = np.mean([m['mae'] for m in all_metrics])
            avg_r2 = np.mean([m['r2'] for m in all_metrics])
            f.write(f"\nOverall Model Performance for {category}:\n")
            f.write(f"Average RMSE: {avg_rmse:.2f}\n")
            f.write(f"Average MAE: {avg_mae:.2f}\n")
            f.write(f"Average R² Score: {avg_r2:.3f}\n")
            f.write("=" * 80 + "\n")
            print(f"Overall Model Performance for {category}: \n\tAverage RMSE: {avg_rmse:.2f} \n\tAverage MAE: {avg_mae:.2f} \n\tAverage R² Score: {avg_r2:.3f}")

def train_category_models(position_type: PositionCategory, start_year: int = 2019, end_year: int = 2024) -> None:
    """
    Train models for stat categories for the specified position type.
    
    Args:
        position_type: PositionCategory to train models for
        start_year: First year of data to use
        end_year: Last year of data to use
    """
    train_and_evaluate_model(position_type, LeagueType.CATEGORIES, start_year, end_year)

def main():
    for position_type in PositionCategory:
        train_and_evaluate_model(position_type, LeagueType.POINTS)
    
    for position_type in PositionCategory:
        train_and_evaluate_model(position_type, LeagueType.CATEGORIES)


if __name__ == "__main__":
    main()