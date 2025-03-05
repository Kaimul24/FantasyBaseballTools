import sys
import os
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DataProcessing.DataProcessingComposition import Predict2025Processor
from DataProcessing.BatterDataProcessing import BatterDataProcessing
import pandas as pd
import xgboost as xgb

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

def predict_points(model: xgb.XGBRegressor, prediction_data: dict) -> pd.DataFrame:
    """
    Use the model to predict fantasy points
    
    Args:
        model: Trained XGBoost model
        prediction_data: Dictionary with processed prediction data
        
    Returns:
        DataFrame with player names and predicted points
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
        f'Predicted_{prediction_year}_Points': predictions
    })
    
    # Sort by predicted points descending
    results = results.sort_values(f'Predicted_{prediction_year}_Points', ascending=False)
    
    return results

def main():
    # Load model
    print("Loading model...")
    model = load_model("batter_model.json")

    # Load and process data from recent years
    print("Getting data...")
    batters = BatterDataProcessing(start_year=2022, end_year=2024)
    batters.filter_data()
    batters.calc_fantasy_points()

    data_processor = Predict2025Processor(batters)
    
    # Prepare data for prediction
    print("Preparing data for 2025 predictions...")
    prediction_data = data_processor.prepare_data()

    # Predict 2025 points
    print("Predicting 2025 points...")
    predictions_2025 = predict_points(model, prediction_data)
    
    # Display top 20 predictions
    print("\nTop 20 projected players for 2025:")
    print(predictions_2025.head(20))
    
    # Save predictions to CSV
    predictions_2025.to_csv("predicted_2025_points2.csv", index=False)
    print("\nPredictions saved to 'predicted_2025_points.csv'")
    
    # Save top 200 for fantasy draft
    top_200 = predictions_2025.head(200)
    top_200.to_csv("2025_fantasy_draft_rankings.csv", index=False)
    print("Top 200 players saved to '2025_fantasy_draft_rankings2.csv'")

if __name__ == "__main__":
    main()
