import sys
import os

from ..FangraphsScraper.fangraphsScraper import PositionCategory
from ..DataProcessing.DataPipelines import PredictionDataPrep
from ..DataProcessing.BatterDataProcessing import BatterDataProcessing
from ..DataProcessing.StarterDataProcessing import StarterDataProcessing
from ..DataProcessing.RelieverDataProcessing import RelieverDataProcessing
import pandas as pd
import xgboost as xgb

from config import PREDICTIONS_DIR

def load_model(filepath: str) -> xgb.XGBRegressor:
    """
    Load a saved XGBoost model from disk
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded XGBoost model
    """
    model = xgb.XGBRegressor()
    model.load_model(filepath)
    print(f"Model successfully loaded from {filepath}")
    return model

def predict_points(model: xgb.XGBRegressor, prediction_data: dict, position_label: PositionCategory) -> pd.DataFrame:
    """
    Use the model to predict fantasy points
    
    Args:
        model: Trained XGBoost model
        prediction_data: Dictionary with processed prediction data
        position_label: Label for the player position ('Batter', 'SP', or 'RP')
        
    Returns:
        DataFrame with player names, position, and predicted points
    """
    weighted_data = prediction_data['weighted_data']
    prediction_year = prediction_data['prediction_year']
    
    # Get feature columns (excluding PlayerName and TotalPoints)
    feature_cols = [col for col in weighted_data.columns 
                   if col != 'PlayerName' and col != 'TotalPoints']
    
    # Prepare data for prediction
    X_pred = weighted_data[feature_cols]
    
    # Predict points
    predictions = model.predict(X_pred)
    
    # Create results dataframe
    results = pd.DataFrame({
        'PlayerName': weighted_data['PlayerName'],
        'Position': position_label,
        f'Predicted_{prediction_year}_Points': predictions
    })
    
    # Sort by predicted points descending
    results = results.sort_values(f'Predicted_{prediction_year}_Points', ascending=False)
    
    return results


def main():
    # Create predictions for each position type
    all_predictions = []
    
    # BATTERS
    print("\n=== BATTER PREDICTIONS ===")
    print("Loading model...")
    batter_model = load_model(os.path.join("models","batter_model.json"))

    print("Getting batter data...")
    batters = BatterDataProcessing(start_year=2022, end_year=2024)
    batters.filter_and_calc_points()

    batter_processor = PredictionDataPrep(batters)
    
    print("Preparing data for 2025 predictions...")
    batter_prediction_data = batter_processor.prepare_data()

    print("Predicting 2025 points...")
    batter_predictions = predict_points(batter_model, batter_prediction_data, "Batter")
    all_predictions.append(batter_predictions)
    
    print(f"\nTop 10 projected batters for 2025:")
    print(batter_predictions.head(10))
    
    # STARTERS
    print("\n=== STARTING PITCHER PREDICTIONS ===")
    print("Loading model...")
    starter_model = load_model(os.path.join("models","starter_model.json"))

    print("Getting starter data...")
    starters = StarterDataProcessing(start_year=2022, end_year=2024)
    starters.filter_and_calc_points()

    starter_processor = PredictionDataPrep(starters)
    
    print("Preparing data for 2025 predictions...")
    starter_prediction_data = starter_processor.prepare_data()

    print("Predicting 2025 points...")
    starter_predictions = predict_points(starter_model, starter_prediction_data, "SP")
    all_predictions.append(starter_predictions)
    
    print(f"\nTop 10 projected starters for 2025:")
    print(starter_predictions.head(10))
    
    # RELIEVERS
    print("\n=== RELIEF PITCHER PREDICTIONS ===")
    print("Loading model...")
    reliever_model = load_model(os.path.join("models","reliever_model.json"))

    print("Getting reliever data...")
    relievers = RelieverDataProcessing(start_year=2022, end_year=2024)
    relievers.filter_and_calc_points()

    reliever_processor = PredictionDataPrep(relievers)
    
    print("Preparing data for 2025 predictions...")
    reliever_prediction_data = reliever_processor.prepare_data()

    print("Predicting 2025 points...")
    reliever_predictions = predict_points(reliever_model, reliever_prediction_data, "RP")
    all_predictions.append(reliever_predictions)
    
    print(f"\nTop 10 projected relievers for 2025:")
    print(reliever_predictions.head(10))
    
    # COMBINE ALL PREDICTIONS
    print("\n=== COMBINING ALL PREDICTIONS ===")
    combined_predictions = pd.concat(all_predictions)
    combined_predictions = combined_predictions.sort_values(f'Predicted_2025_Points', ascending=False)
    
    # Display top 20 overall predictions
    print("\nTop 20 projected players overall for 2025:")
    print(combined_predictions.head(20))
    
    # Save position-specific predictions to CSV
    batter_predictions.to_csv((PREDICTIONS_DIR / "predicted_2025_batters.csv"), index=False)
    starter_predictions.to_csv((PREDICTIONS_DIR / "predicted_2025_starters.csv"), index=False)
    reliever_predictions.to_csv((PREDICTIONS_DIR / "predicted_2025_relievers.csv"), index=False)
    
    # Save combined predictions to CSV
    combined_predictions.to_csv((PREDICTIONS_DIR / "predicted_2025_all.csv"), index=False)
    print("\nPredictions saved to CSV files:")
    print("- predicted_2025_batters.csv")
    print("- predicted_2025_starters.csv")
    print("- predicted_2025_relievers.csv")
    print("- predicted_2025_all_positions.csv")
    

if __name__ == "__main__":
    main()
