from DataProcessing import DataProcessing
from FangraphsScraper.fangraphsScraper import PositionCategory
import pandas as pd
import numpy as np
from typing import List, TypedDict
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score  # Add this import at the top

### EXCLUDE 2020

class DatasetSplit(TypedDict):
    train_years: List[int]
    val_year: int
    test_year: int
    train_data: pd.DataFrame
    val_data: pd.DataFrame
    test_data: pd.DataFrame

class WeightedDatasetSplit(DatasetSplit):
    weighted_data: pd.DataFrame


def load_data() -> pd.DataFrame:
    """Load initial data"""
    df = DataProcessing(PositionCategory.BATTER)
    df.filter_data()
    df.calc_fantasy_points()
    return df.data 

def create_rolling_datasets(df: pd.DataFrame) -> List[DatasetSplit]:
    """Create rolling train/validation/test splits"""
    # Get all years from column names and sort them
    years = sorted(list(set(
        int(col.split('_')[0]) 
        for col in df.columns 
        if '_' in col and col.split('_')[0].isdigit()
    )))
    
    # Remove 2020 if present
    years = [year for year in years if year != 2020]
    
    # Create windows: each window needs 2 training years, 1 validation year, and 1 test year
    windows = []
    for i in range(len(years) - 3):  # -3 because we need 4 years for each window
        window = {
            'train_years': [years[i], years[i+1]],
            'val_year': years[i+2],               
            'test_year': years[i+3]  
        }
        windows.append(window)
    
    datasets = []
    
    for window in windows:
        train_years = window['train_years']
        val_year = window['val_year']
        test_year = window['test_year']
        
        # Get columns for each set
        train_cols = [col for col in df.columns if any(str(year) in col for year in train_years)]
        val_cols = [col for col in df.columns if str(val_year) in col]
        test_cols = [col for col in df.columns if str(test_year) in col]
        
        if test_cols:  # Only create dataset if test year exists
            dataset = {
                'train_years': train_years,
                'val_year': val_year,
                'test_year': test_year,
                'train_data': df[['PlayerName'] + train_cols].copy(),
                'val_data': df[['PlayerName'] + val_cols].copy(),
                'test_data': df[['PlayerName'] + test_cols].copy()
            }
            datasets.append(dataset)
    
    return datasets

def concat_training_windows(datasets: List[DatasetSplit]) -> List[WeightedDatasetSplit]:
    """
    Process each training window separately applying weights to years.
    For 2 years: 60% recent year, 40% older year
    Returns list of processed training windows.
    """
    processed_windows = []
    
    for dataset in datasets:
        train_data = dataset['train_data'].copy()
        train_years = sorted(dataset['train_years'])
        
        weighted_data = pd.DataFrame({'PlayerName': train_data['PlayerName'].unique()})
        
        year_prefix = f"{train_years[-1]}_"
        base_stats = [col.replace(year_prefix, '') for col in train_data.columns 
                     if year_prefix in col]
        
        for stat in base_stats:
            year2_col = f"{train_years[1]}_{stat}"  # More recent year
            year1_col = f"{train_years[0]}_{stat}"  # Older year
            
            stat_data = pd.DataFrame()
            for year_col in [year1_col, year2_col]:
                if year_col in train_data.columns:
                    year_stats = train_data[['PlayerName', year_col]].copy()
                    stat_data = pd.merge(stat_data, year_stats, on='PlayerName', how='outer') if not stat_data.empty else year_stats

            weighted_stat = pd.Series(0, index=stat_data.index)
            
            # Set weights: 60% recent year, 40% older year
            weights = {year2_col: 0.6, year1_col: 0.4}
            
            # Apply weights
            for year_col, weight in weights.items():
                if year_col in stat_data.columns:
                    weighted_stat += stat_data[year_col].fillna(0) * weight
                
            weighted_data[stat] = weighted_stat
        
        processed_windows.append({
            'train_years': train_years,
            'val_year': dataset['val_year'],
            'test_year': dataset['test_year'],
            'weighted_data': weighted_data,
            'train_data': train_data,  # Keep original training data for reference
            'val_data': dataset['val_data'],
            'test_data': dataset['test_data'],
        })
    
    return processed_windows

def remove_stats(windows: List[WeightedDatasetSplit]) -> List[WeightedDatasetSplit]:
    '''Remove total points and counting stats from training data'''
    for window in windows:
        weighted_data = window['weighted_data']
        cols_to_drop = ['1B', '2B', '3B', 'HR', 'R', 'RBI', 
                        'HBP', 'SB']

        weighted_data = weighted_data.drop(cols_to_drop, axis=1)
        window['weighted_data'] = weighted_data

    return windows

def remove_year_prefixes(windows: List[WeightedDatasetSplit]) -> List[WeightedDatasetSplit]:
    '''Remove year prefixes from validation and test data columns'''
    for window in windows:
        # Handle validation data
        val_year = str(window['val_year'])
        val_cols = window['val_data'].columns
        val_rename = {
            col: col.replace(f"{val_year}_", "") 
            for col in val_cols 
            if col.startswith(f"{val_year}_")
        }
        window['val_data'] = window['val_data'].rename(columns=val_rename)
        
        # Handle test data
        test_year = str(window['test_year'])
        test_cols = window['test_data'].columns
        test_rename = {
            col: col.replace(f"{test_year}_", "") 
            for col in test_cols 
            if col.startswith(f"{test_year}_")
        }
        window['test_data'] = window['test_data'].rename(columns=test_rename)
        
        # Also remove 'weighted_' prefix from weighted_data
        weighted_cols = window['weighted_data'].columns
        weighted_rename = {
            col: col.replace("weighted_", "") 
            for col in weighted_cols 
            if col.startswith("weighted_") and col != "PlayerName"
        }
        window['weighted_data'] = window['weighted_data'].rename(columns=weighted_rename)
    
    return windows

def train_model(windows: List[WeightedDatasetSplit]) -> List[WeightedDatasetSplit]:
    '''Train single XGBoost model and predict on validation data'''
    non_valid_cols = ['1B', '2B', '3B', 'HR', 'R', 'RBI', 'HBP', 'SB']
    
    # Create single model instance
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        objective='reg:squarederror',
        random_state=42
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
    
    # Make predictions on validation data for each window
    for window in windows:
        val_data = window['val_data']
        val_year = window['val_year']
        
        # Get validation features (removing year prefix)
        val_features = [col for col in val_data.columns if col != 'PlayerName' and col != 'TotalPoints' and col not in non_valid_cols]
        
        if not val_features:
            print(f"Skipping predictions for validation year {val_year}: no features found")
            continue
            
        X_val = val_data[val_features]
        y_val = val_data["TotalPoints"]
        
        val_predictions = model.predict(X_val)
        
        window['model'] = model
        window['predictions'] = pd.DataFrame({
            'PlayerName': val_data['PlayerName'],
            'predicted_weighted_TotalPoints': val_predictions,
            'actual_weighted_TotalPoints': y_val.values
        })
    
    return windows

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate RMSE and R² score for predictions
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
    
    Returns:
        Dictionary containing RMSE and R² score
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
    r2 = r2_score(y_true, y_pred)
    
    return {
        'rmse': rmse,
        'r2': r2
    }

def main():
    # Load data
    print("Loading data...\n")
    df = load_data()
    
    # Create rolling datasets
    print("Creating datasets...\n")
    datasets = create_rolling_datasets(df)
    print(f"Number of datasets: {len(datasets)}\n")

    # Get processed training windows
    print("Processing training windows...\n")
    processed_windows = concat_training_windows(datasets)

    '''
    Each window has the following keys:
        window['train_years']
        window['val_year']
        window['test_year']
        window['weighted_data']
        window['train_data']
        window['val_data']
        window['test_data']
    '''

    # Remove year prefixes
    print("Removing year prefixes...\n")
    processed_windows = remove_year_prefixes(processed_windows)

    # Remove counting stats from training data
    print("Removing counting stats from training data...\n")
    processed_windows = remove_stats(processed_windows)

    #  Print all data to file
    with open('all_data.txt', 'w') as f:
        for i, window in enumerate(processed_windows):
            f.write(f"\n{'='*80}\n")
            f.write(f"Window {i+1}:\n")
            f.write(f"Training years: {window['train_years']}\n")
            f.write(f"Validation year: {window['val_year']}\n")
            f.write(f"Test year: {window['test_year']}\n")
            
            f.write("\nWeighted Training Data:\n")
            f.write(f"Shape: {window['weighted_data'].shape}\n")
            f.write(window['weighted_data'].to_string())
            
            f.write("\n\nValidation Data:\n")
            f.write(f"Shape: {window['val_data'].shape}\n")
            f.write(window['val_data'].to_string())
            
            f.write("\n\nTest Data:\n")
            f.write(f"Shape: {window['test_data'].shape}\n")
            f.write(window['test_data'].to_string())
            f.write(f"\n{'='*80}\n")

    # Train model for each window
    print("Training models...\n")
    trained_windows = train_model(processed_windows)

    with open("predictions.txt", "w") as f:
        # Track overall metrics
        all_metrics = []
        
        for i, window in enumerate(trained_windows):
            if "predictions" not in window:
                f.write(f"Window {i+1}: No predictions available.\n")
                continue
            
            f.write(f"Window {i+1}:\n")
            f.write(f"Training years: {window['train_years']}, Validation year: {window['val_year']}, Test year: {window['test_year']}\n")
            predictions = window["predictions"].copy()
            
            # Calculate metrics
            metrics = calculate_metrics(
                predictions['actual_weighted_TotalPoints'],
                predictions['predicted_weighted_TotalPoints']
            )
            all_metrics.append(metrics)
            
            # Write metrics
            f.write("\nValidation Metrics:\n")
            f.write(f"RMSE: {metrics['rmse']:.2f}\n")
            f.write(f"R² Score: {metrics['r2']:.3f}\n\n")
            
            # Compute and write predictions
            predictions["percent_diff"] = predictions.apply(
                lambda row: abs(row["predicted_weighted_TotalPoints"] - row["actual_weighted_TotalPoints"]) / row["actual_weighted_TotalPoints"] * 100
                            if row["actual_weighted_TotalPoints"] != 0 else 0, axis=1)
            
            f.write(f"{'PlayerName':<20}{'Predicted':>15}{'Actual':>15}{'Percent Diff (%)':>20}\n")
            f.write("-" * 70 + "\n")
            
            for _, row in predictions.iterrows():
                f.write(f"{row['PlayerName']:<20}{row['predicted_weighted_TotalPoints']:>15.2f}"
                        f"{row['actual_weighted_TotalPoints']:>15.2f}{row['percent_diff']:>20.2f}\n")
            f.write("\n" + "=" * 80 + "\n\n")
        
        # Write average metrics across all windows
        if all_metrics:
            avg_rmse = np.mean([m['rmse'] for m in all_metrics])
            avg_r2 = np.mean([m['r2'] for m in all_metrics])
            f.write("\nOverall Model Performance:\n")
            f.write(f"Average RMSE: {avg_rmse:.2f}\n")
            f.write(f"Average R² Score: {avg_r2:.3f}\n")
            f.write("=" * 80 + "\n")

if __name__ == "__main__":
    main()