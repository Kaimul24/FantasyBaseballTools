import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DataProcessing.DataPipelines import TrainingDataPrep, LeagueType
from DataProcessing.BatterDataProcessing import BatterDataProcessing
from PointsPredictors.PointsPredictor3 import Model
import pandas as pd
import numpy as np


data = BatterDataProcessing()
data.filter_data()

with (open ("Category_test.txt" , "w")) as f:
    f.write(data.data.to_string())

data_processor = TrainingDataPrep(data, league_type=LeagueType.CATEGORIES)
windows = data_processor.prepare_data()

with open(('Category_batter_data.txt'), 'w') as f:
        for i, window in enumerate(windows):
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

col = 'HR'

batter_model = Model(data_processor, league_type=LeagueType.CATEGORIES, category=col)
model = batter_model.train_model(windows, tune_hyperparams=True)

tested_windows = batter_model.predict_model(model, windows)

with open(("Category_batter_predictions.txt"), "w") as f:
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
            predictions[f'actual_{col}'],
            predictions[f'predicted_{col}']
        )
        all_metrics.append(metrics)
        
        # Write metrics
        f.write("\nTest Metrics:\n")
        f.write(f"RMSE: {metrics['rmse']:.2f}\n")
        f.write(f"MAE: {metrics['mae']:.2f}\n")
        f.write(f"R² Score: {metrics['r2']:.3f}\n\n")
        
        # Compute and write predictions
        predictions["percent_diff"] = predictions.apply(
            lambda row: abs(row[f'predicted_{col}'] - row[f'actual_{col}']) / row[f'actual_{col}'] * 100
                        if row[f'actual_{col}'] != 0 else 0, axis=1)
        
        f.write(f"{'PlayerName':<20}{'Predicted':>15}{'Actual':>15}{'Percent Diff (%)':>20}\n")
        f.write("-" * 70 + "\n")
        
        for _, row in predictions.iterrows():
            f.write(f"{row['PlayerName']:<20}{row[f'predicted_{col}']:>15.2f}"
                    f"{row[f'actual_{col}']:>15.2f}{row['percent_diff']:>20.2f}\n")
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
