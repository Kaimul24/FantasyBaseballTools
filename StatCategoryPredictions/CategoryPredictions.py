import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
stats_category_models = "stats_category_models" 
os.makedirs(stats_category_models, exist_ok=True)


from PointsPredictors.PointsPredictor3 import Model
from DataProcessing.DataPipelines import TrainingDataPrep, LeagueType, BaseDataPrep
from DataProcessing.BatterDataProcessing import BatterDataProcessing
from DataProcessing.StarterDataProcessing import StarterDataProcessing
from DataProcessing.RelieverDataProcessing import RelieverDataProcessing
from DataProcessing.DataProcessing import WeightedDatasetSplit, DataProcessing
from FangraphsScraper.fangraphsScraper import PositionCategory 
import pandas as pd
import numpy as np

class CategoryPredictions:
    def __init__(self, position_category: PositionCategory, start_year: int = 2019, end_year: int = 2024):
        self.position_category = position_category
            
        if self.position_category == PositionCategory.BATTER:
            self.data = BatterDataProcessing(start_year=start_year, end_year=end_year)
        elif self.position_category == PositionCategory.SP:
            self.data = StarterDataProcessing(start_year=start_year, end_year=end_year)
        elif self.position_category == PositionCategory.RP:
            self.data = RelieverDataProcessing(start_year=start_year, end_year=end_year)
        else:
            raise TypeError("Unknown position category")
        
        
        
    def train_categories(self):
        data = self.data
        data.filter_data()
        data_processor = TrainingDataPrep(data, league_type=LeagueType.CATEGORIES)
        windows = data_processor.prepare_data()

        for category in self.data.stat_categories:
            model = Model(data_processor, league_type=LeagueType.CATEGORIES, category=category)
            trained_model = model.train_model(windows)

            Model.save_trained_model(trained_model, os.path.join(stats_category_models,f"batter_{category}_model.json"))
            tested_windows = model.predict_model(trained_model, windows)

            with open(("category_batter_predictions.txt"), "w") as f:
             # Track overall metrics
                all_metrics = []
                
                for i, window in enumerate(tested_windows):
                    if "predictions" not in window:
                        f.write(f"Window {i+1}: No predictions available.\n")
                        continue
                    f.write(f"Stat: {category}")
                    f.write(f"Window {i+1}:\n")
                    f.write(f"Training years: {window['train_years']}, Test year: {window['test_year']}\n")
                    predictions = window["predictions"].copy()
                    
                    # Calculate metrics
                    metrics =  Model.calculate_metrics(
                        predictions[f'actual_{category}'],
                        predictions[f'predicted_{category}']
                    )
                    all_metrics.append(metrics)
                    
                    # Write metrics
                    f.write("\nTest Metrics:\n")
                    f.write(f"RMSE: {metrics['rmse']:.2f}\n")
                    f.write(f"MAE: {metrics['mae']:.2f}\n")
                    f.write(f"R² Score: {metrics['r2']:.3f}\n\n")
                    
                    # Compute and write predictions
                    predictions["percent_diff"] = predictions.apply(
                        lambda row: abs(row[f'predicted_{category}'] - row[f'actual_{category}']) / row[f'actual_{category}'] * 100
                                    if row[f'actual_{category}'] != 0 else 0, axis=1)
                    
                    f.write(f"{'PlayerName':<20}{'Predicted':>15}{'Actual':>15}{'Percent Diff (%)':>20}\n")
                    f.write("-" * 70 + "\n")
                    
                    for _, row in predictions.iterrows():
                        f.write(f"{row['PlayerName']:<20}{row[f'predicted_{category}']:>15.2f}"
                                f"{row[f'actual_{category}']:>15.2f}{row['percent_diff']:>20.2f}\n")
                    f.write("\n" + "=" * 80 + "\n\n")
                
                # Write average metrics across all windows
                if all_metrics:
                    avg_rmse = np.mean([m['rmse'] for m in all_metrics])
                    avg_mae = np.mean([m['mae'] for m in all_metrics])
                    avg_r2 = np.mean([m['r2'] for m in all_metrics])
                    f.write(f"\nOverall Model Performance for {category}:\n")
                    f.write(f"Average RMSE: {avg_rmse:.2f}\n")
                    f.write(f"Average MAE: {avg_mae:.2f}\n")
                    f.write(f"Average R² Score: {avg_r2:.3f}\n")
                    f.write("=" * 80 + "\n")
                    print(f"Overall Model Performance: \n\tAverage RMSE: {avg_rmse:.2f} \n\tAverage MAE: {avg_mae:.2f} \n\tAverage R² Score: {avg_r2:.3f}")
    

batter_cats = CategoryPredictions(PositionCategory.BATTER)
batter_cats.run()

