from ..FangraphsScraper.fangraphsScraper import PositionCategory
from ..DataProcessing.DataPipelines import PredictionDataPrep
from ..DataProcessing.DataProcessing import LeagueType
from ..DataProcessing.BatterDataProcessing import BatterDataProcessing
from ..DataProcessing.StarterDataProcessing import StarterDataProcessing
from ..DataProcessing.RelieverDataProcessing import RelieverDataProcessing
import pandas as pd
import xgboost as xgb
from typing import Dict, Union

from config import PREDICTIONS_DIR, MODELS_DIR

class FuturePredictions:

    def __init__(self, league_type: LeagueType, position_type: PositionCategory, start_year: int = 2022, end_year: int = 2024):
        """
        Initialize FuturePredictions with league type, position type, and year range.
        
        Args:
            league_type: Type of league (POINTS or CATEGORIES)
            position_type: Type of player position (BATTER, SP, or RP)
            start_year: First year of data to use for predictions
            end_year: Last year of data to use for predictions
        """
        self.league_type = league_type
        self.position_type = position_type
        self.start_year = start_year
        self.end_year = end_year
        self.prediction_data = self.prepare_data()

    def _load_model(self, model_name: str) -> None:
        """
        Load a saved XGBoost model from disk.
        
        Args:
            model_name: Name of the saved model file
        """
        self.model = xgb.XGBRegressor()
        self.model.load_model(MODELS_DIR / model_name)
        print(f"Model successfully loaded from {model_name}")
    
    def prepare_data(self) -> Dict:
        """
        Prepare data for making predictions by processing historical data.
        
        Returns:
            Dictionary containing processed prediction data
        """
        position_map = {
            PositionCategory.BATTER: BatterDataProcessing,
            PositionCategory.SP: StarterDataProcessing,
            PositionCategory.RP: RelieverDataProcessing
        }

        if self.position_type not in position_map:
            raise ValueError(f"Invalid position type: {self.position_type}. Must be one of {list(position_map.keys())}")

        data_processor = position_map[self.position_type](league_type=self.position_type, start_year=self.start_year, end_year=self.end_year)
        
        if self.league_type == LeagueType.POINTS:
            data_processor.filter_and_calc_points()
            self.invalid_cols = data_processor.get_counting_stats()
            self.stat_categories = []
        else:
            data_processor.filter_data()
            self.invalid_cols = []
            self.stat_categories = data_processor.stat_categories
        
        data_prep = PredictionDataPrep(data_processor)
        processed_data = data_prep.prepare_data()

        return processed_data
    
    def predict_points(self) -> pd.DataFrame:
        """
        Predict fantasy points for players.
        
        Returns:
            DataFrame with player names, positions, and predicted points
        """
        return self._predict_future()
    
    def predict_categories(self) -> Dict[str, pd.DataFrame]:
        """
        Predict stat categories for players.
        
        Returns:
            Dictionary mapping category names to DataFrames with predictions
        """
        all_predictions = {}
        for category in self.stat_categories:
            all_predictions[category] = self._predict_future(category)

        return all_predictions

    def _predict_future(self, target: str = "TotalPoints") -> pd.DataFrame:
        """
        Use the model to predict future performance for specified target.
        
        Args:
            target: Target metric to predict (default: "TotalPoints")
            
        Returns:
            DataFrame with player names, position, and predicted values
        """
        if self.league_type == LeagueType.POINTS:
            self._load_model(f"{str(self.position_type)[17:]}_model.json")
        else:
            self._load_model(f"{str(self.position_type)[17:].lower()}_{target}_model.json")

        weighted_data = self.prediction_data['weighted_data']
        prediction_year = self.prediction_data['prediction_year']
        
        feature_cols = [col for col in weighted_data.columns 
                    if col != 'PlayerName' and col != f"{target}"
                    and col not in self.invalid_cols]
        
        X_pred = weighted_data[feature_cols]

        if self.league_type == LeagueType.CATEGORIES:
                if self.position_type == PositionCategory.BATTER:
                    if target == 'TB':
                        X_pred = X_pred.drop(['1B', '2B', '3B', 'HR'], axis=1)
                    elif target == 'NSB':
                        X_pred = X_pred.drop(['SB', 'CS'], axis=1)
                    elif target == 'OPS':
                        X_pred = X_pred.drop(['OBP', 'SLG'], axis=1)
                else:
                    if target == 'SO':
                        X_pred = X_pred.drop(['SO'], axis=1, errors='ignore')
                    elif target == 'WHIP':
                        X_pred = X_pred.drop(['WHIP'], axis=1, errors='ignore')
                    elif target == 'ERA':
                        X_pred = X_pred.drop(['ERA'], axis=1, errors='ignore')
                    elif target== 'QS':
                        X_pred = X_pred.drop(['QS'], axis=1, errors='ignore')
                    elif target == 'NSVH':
                        X_pred = X_pred.drop(['SV', 'HLD', 'BS'], axis=1)

        predictions = self.model.predict(X_pred)
        
        results = pd.DataFrame({
            'PlayerName': weighted_data['PlayerName'],
            'Position': f'{str(self.position_type)[17:]}',
            f'Predicted_{prediction_year}_{target}': predictions
        })
        
        results = results.sort_values(f'Predicted_{prediction_year}_{target}', ascending=False)
        
        return results
    
    def save_predictions(self, predictions: Union[pd.DataFrame, Dict[str, pd.DataFrame]], year: int = 2025) -> None:
        """
        Save predictions to CSV files in the predictions directory.
        
        Args:
            predictions: DataFrame (for points) or dictionary of DataFrames (for categories)
            year: The year for which predictions are made (default: 2025)
        """
        position_name = str(self.position_type)[17:].lower()
        
        if isinstance(predictions, pd.DataFrame):
            file_path = PREDICTIONS_DIR / f"predicted_{year}_{position_name}_points.csv"
            predictions.to_csv(file_path, index=False)
            print(f"Points predictions saved to: {file_path}")
        
        elif isinstance(predictions, dict):
            for category, df in predictions.items():
                file_path = PREDICTIONS_DIR / f"predicted_{year}_{position_name}_{category}.csv"
                df.to_csv(file_path, index=False)
                print(f"{category} predictions saved to: {file_path}")
        
        else:
            raise TypeError("Predictions must be a DataFrame or dictionary of DataFrames")
        
def main():
    # Create predictions for each position type
    battersPoints2025 = FuturePredictions(league_type=LeagueType.POINTS, position_type=PositionCategory.BATTER)
    battersPointsPredictions = battersPoints2025.predict_points()
    battersPoints2025.save_predictions(battersPointsPredictions)
    
    print(f"\nTop 10 projected batters for 2025:")
    print(battersPointsPredictions.head(10))

    batterCats2025 = FuturePredictions(league_type=LeagueType.CATEGORIES, position_type=PositionCategory.BATTER)
    batterCatsPredictions = batterCats2025.predict_categories()
    batterCats2025.save_predictions(batterCatsPredictions)
    
    for category, _ in batterCatsPredictions.items():
        print(f"\nTop 10 projected {category} for 2025:")
        print(batterCatsPredictions[category].head(10))
        
    startersPoints2025 = FuturePredictions(league_type=LeagueType.POINTS, position_type=PositionCategory.SP)
    startersPointsPredictions = startersPoints2025.predict_points()
    startersPoints2025.save_predictions(startersPointsPredictions)
    
    print(f"\nTop 10 projected starting pitchers for 2025:")
    print(startersPointsPredictions.head(10))
    
    starterCats2025 = FuturePredictions(league_type=LeagueType.CATEGORIES, position_type=PositionCategory.SP)
    starterCatsPredictions = starterCats2025.predict_categories()
    starterCats2025.save_predictions(starterCatsPredictions)
    
    relieversPoints2025 = FuturePredictions(league_type=LeagueType.POINTS, position_type=PositionCategory.RP)
    relieversPointsPredictions = relieversPoints2025.predict_points()
    relieversPoints2025.save_predictions(relieversPointsPredictions)
    
    print(f"\nTop 10 projected relief pitchers for 2025:")
    print(relieversPointsPredictions.head(10))
    
    relieverCats2025 = FuturePredictions(league_type=LeagueType.CATEGORIES, position_type=PositionCategory.RP)
    relieverCatsPredictions = relieverCats2025.predict_categories()
    relieverCats2025.save_predictions(relieverCatsPredictions)
    
    all_points_predictions = pd.concat([
        battersPointsPredictions,
        startersPointsPredictions,
        relieversPointsPredictions
    ])
    
    all_points_predictions = all_points_predictions.sort_values(
        f'Predicted_2025_TotalPoints', ascending=False
    )
    
    all_predictions_path = PREDICTIONS_DIR / "predicted_2025_all_players.csv"
    all_points_predictions.to_csv(all_predictions_path, index=False)
    print(f"Combined predictions saved to: {all_predictions_path}")
    

if __name__ == "__main__":
    main()
