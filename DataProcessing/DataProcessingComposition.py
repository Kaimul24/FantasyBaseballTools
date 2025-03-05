from FangraphsScraper.fangraphsScraper import FangraphsScraper, PositionCategory
import pandas as pd
from typing import List
from DataProcessing.DataProcessing import (
    DatasetSplit, WeightedDatasetSplit, PredictionDataset,
)
from DataProcessing.BatterDataProcessing import BatterDataProcessing
from DataProcessing.StarterDataProcessing import StarterDataProcessing
from DataProcessing.RelieverDataProcessing import RelieverDataProcessing

class TrainingDataProcessor:
    """
    Class for processing data for model training using composition instead of inheritance.
    Includes functions for creating rolling datasets, applying weights, and preprocessing data.
    """
    def __init__(self, data_processor):
        """
        Initialize with a data processor instance (Batter, Starter, or Reliever)
        
        Args:
            data_processor: An instance of BatterDataProcessing, StarterDataProcessing, or RelieverDataProcessing
        """
        self.data_processor = data_processor
        self.data = data_processor.data
        
    def create_datasets(self) -> List[DatasetSplit]:
        """Create rolling train/validation/test splits"""
        print("Creating training datasets...\n")
        
        # Get all years from column names and sort them
        years = sorted(list(set(
            int(col.split('_')[0]) 
            for col in self.data.columns 
            if '_' in col and col.split('_')[0].isdigit()
        )))
        
        # Remove 2020 if present
        years = [year for year in years if year != 2020]
        print(f"Available years: {years}")
        
        # Create windows: each window needs 3 training years, 1 test year
        windows = []
        for i in range(len(years) - 3):  # -3 because we need 4 years for each window
            window = {
                'train_years': [years[i], years[i+1], years[i+2]],           
                'test_year': years[i+3]  
            }
            windows.append(window)
        
        datasets = []
        
        for window in windows:
            train_years = window['train_years']
            test_year = window['test_year']
            
            # Get columns for each set
            train_cols = [col for col in self.data.columns if any(str(year) in col for year in train_years)]
            test_cols = [col for col in self.data.columns if str(test_year) in col]
            
            if test_cols:  # Only create dataset if test year exists
                dataset = {
                    'train_years': train_years,
                    'test_year': test_year,
                    'train_data': self.data[['PlayerName'] + train_cols].copy(),
                    'test_data': self.data[['PlayerName'] + test_cols].copy()
                }
                datasets.append(dataset)
        
        print(f"Created {len(datasets)} training datasets\n")
        return datasets

    def apply_weights(self, datasets: List[DatasetSplit]) -> List[WeightedDatasetSplit]:
        """
        Process each training window separately applying weights to years.
        For 3 years: 50% recent year, 30% recent year - 1, 20% recent year - 2.
        Returns list of processed training windows.
        """
        print("Applying weights to training data...\n")
        
        processed_windows = []
        
        for dataset in datasets:
            train_data = dataset['train_data'].copy()
            train_years = sorted(dataset['train_years'])
            
            weighted_data = pd.DataFrame({'PlayerName': train_data['PlayerName'].unique()})
            
            year_prefix = f"{train_years[-1]}_"
            base_stats = [col.replace(year_prefix, '') for col in train_data.columns 
                         if year_prefix in col]
            
            for stat in base_stats:
                year3_col = f"{train_years[2]}_{stat}"  # Recent year
                year2_col = f"{train_years[1]}_{stat}"  # Recent year - 1
                year1_col = f"{train_years[0]}_{stat}"  # Recent year - 2
                
                stat_data = pd.DataFrame()
                for year_col in [year1_col, year2_col, year3_col]:
                    if year_col in train_data.columns:
                        year_stats = train_data[['PlayerName', year_col]].copy()
                        stat_data = pd.merge(stat_data, year_stats, on='PlayerName', how='outer') if not stat_data.empty else year_stats

                weighted_stat = pd.Series(0, index=stat_data.index)
                
                # Set weights: 50% recent year, 30% recent-1, 20% recent-2
                weights = {year3_col: 0.5, year2_col: 0.3, year1_col: 0.2}
                
                # Apply weights
                for year_col, weight in weights.items():
                    if year_col in stat_data.columns:
                        weighted_stat += stat_data[year_col].fillna(0) * weight
                    
                weighted_data[stat] = weighted_stat
            
            processed_windows.append({
                'train_years': train_years,
                'test_year': dataset['test_year'],
                'weighted_data': weighted_data,
                'train_data': train_data, 
                'test_data': dataset['test_data'],
            })
        
        return processed_windows

    def remove_counting_stats(self, windows: List[WeightedDatasetSplit]) -> List[WeightedDatasetSplit]:
        '''Remove counting stats from training data'''
        print("Removing counting stats from training data...\n")
        
        # Get counting stats from the underlying data processor
        cols_to_drop = self.data_processor.get_counting_stats()
        
        for window in windows:
            weighted_data = window['weighted_data']
            # Drop only columns that actually exist
            cols_to_remove = [col for col in cols_to_drop if col in weighted_data.columns]
            if cols_to_remove:
                weighted_data = weighted_data.drop(cols_to_remove, axis=1)
            window['weighted_data'] = weighted_data

        return windows

    def format_column_names(self, windows: List[WeightedDatasetSplit]) -> List[WeightedDatasetSplit]:
        '''Remove year prefixes from test data columns'''
        print("Formatting column names...\n")
        
        for window in windows:
            # Handle test data
            test_year = str(window['test_year'])
            test_cols = window['test_data'].columns
            test_rename = {
                col: col.replace(f"{test_year}_", "") 
                for col in test_cols 
                if col.startswith(f"{test_year}_")
            }
            window['test_data'] = window['test_data'].rename(columns=test_rename)
            
            # Also remove 'weighted_' prefix from weighted_data if present
            weighted_cols = window['weighted_data'].columns
            weighted_rename = {
                col: col.replace("weighted_", "") 
                for col in weighted_cols 
                if col.startswith("weighted_") and col != "PlayerName"
            }
            window['weighted_data'] = window['weighted_data'].rename(columns=weighted_rename)
        
        return windows

    def prepare_data(self) -> List[WeightedDatasetSplit]:
        """
        Prepare data for model training by creating datasets, applying weights, and preprocessing.
        Returns a list of preprocessed training windows ready for model training.
        """
        datasets = self.create_datasets()
        processed_windows = self.apply_weights(datasets)
        processed_windows = self.format_column_names(processed_windows)
        processed_windows = self.remove_counting_stats(processed_windows)
        
        return processed_windows


class Predict2025Processor:
    """
    Class for processing data to predict 2025 fantasy points using composition.
    Uses the most recent 3 years data to make predictions.
    """
    def __init__(self, data_processor):
        """
        Initialize with a data processor instance (Batter, Starter, or Reliever)
        
        Args:
            data_processor: An instance of BatterDataProcessing, StarterDataProcessing, or RelieverDataProcessing
        """
        self.data_processor = data_processor
        self.data = data_processor.data

    def create_dataset(self) -> PredictionDataset:
        """Create a single prediction dataset using recent years"""
        print("Creating prediction dataset...\n")
        
        # Get years from data
        years = sorted([int(col.split('_')[0]) 
                       for col in self.data.columns 
                       if '_' in col and col.split('_')[0].isdigit()])
        
        # Take the 3 most recent years
        recent_years = sorted(years)[-3:]
        
        if len(recent_years) < 3:
            print(f"Warning: Less than 3 years available: {recent_years}")
        
        print(f"Using years {recent_years} for prediction")
        
        # Create prediction window with recent years as training
        prediction_window = {
            'train_years': recent_years,
            'prediction_year': max(recent_years) + 1
        }
        
        # Get all data columns for training years
        train_cols = [col for col in self.data.columns 
                     if any(str(year) in col for year in prediction_window['train_years'])]
        
        # Create prediction dataset structure
        prediction_data = {
            'train_years': prediction_window['train_years'],
            'prediction_year': prediction_window['prediction_year'],
            'train_data': self.data[['PlayerName'] + train_cols].copy()
        }
        
        print(f"Created prediction dataset for {prediction_window['prediction_year']}\n")
        return prediction_data

    def apply_weights(self, prediction_data: PredictionDataset) -> PredictionDataset:
        """
        Apply weights to the prediction dataset.
        Uses recent years with weights: 50% most recent, 30% second most recent, 20% third most recent.
        """
        print("Applying weights to prediction data...\n")
        
        train_data = prediction_data['train_data'].copy()
        train_years = sorted(prediction_data['train_years'])
        
        weighted_data = pd.DataFrame({'PlayerName': train_data['PlayerName'].unique()})
        
        year_prefix = f"{train_years[-1]}_"
        base_stats = [col.replace(year_prefix, '') for col in train_data.columns 
                     if year_prefix in col]
        
        for stat in base_stats:
            year3_col = f"{train_years[2]}_{stat}" if len(train_years) >= 3 else None  # Most recent
            year2_col = f"{train_years[1]}_{stat}" if len(train_years) >= 2 else None  # Second most recent
            year1_col = f"{train_years[0]}_{stat}" if len(train_years) >= 1 else None  # Third most recent
            
            stat_data = pd.DataFrame()
            for year_col in [year1_col, year2_col, year3_col]:
                if year_col and year_col in train_data.columns:
                    year_stats = train_data[['PlayerName', year_col]].copy()
                    stat_data = pd.merge(stat_data, year_stats, on='PlayerName', how='outer') if not stat_data.empty else year_stats

            weighted_stat = pd.Series(0, index=stat_data.index)
            
            # Set weights: 50% most recent, 30% second most recent, 20% third most recent
            weights = {}
            if year3_col: weights[year3_col] = 0.5
            if year2_col: weights[year2_col] = 0.3
            if year1_col: weights[year1_col] = 0.2
            
            # Normalize weights if some years are missing
            if weights:
                total_weight = sum(weights.values())
                weights = {k: v/total_weight for k, v in weights.items()}
            
            # Apply weights
            for year_col, weight in weights.items():
                if year_col in stat_data.columns:
                    weighted_stat += stat_data[year_col].fillna(0) * weight
                
            weighted_data[stat] = weighted_stat
        
        prediction_data['weighted_data'] = weighted_data
        return prediction_data

    def remove_counting_stats(self, prediction_data: PredictionDataset) -> PredictionDataset:
        """Remove counting stats from weighted data"""
        print("Removing counting stats from prediction data...\n")
        
        # Use the get_counting_stats method from the underlying data processor
        cols_to_drop = self.data_processor.get_counting_stats()
        
        weighted_data = prediction_data['weighted_data']
        
        # Drop only columns that actually exist
        cols_to_remove = [col for col in cols_to_drop if col in weighted_data.columns]
        if cols_to_remove:
            weighted_data = weighted_data.drop(cols_to_remove, axis=1)
                
        prediction_data['weighted_data'] = weighted_data
        return prediction_data

    def format_column_names(self, prediction_data: PredictionDataset) -> PredictionDataset:
        """Format column names by removing prefixes"""
        # For prediction data, we don't need to modify column names
        return prediction_data

    def prepare_data(self) -> PredictionDataset:
        """
        Prepare data for making 2025 predictions.
        Returns a dictionary with processed data ready for prediction.
        """
        prediction_data = self.create_dataset()
        prediction_data = self.apply_weights(prediction_data)
        prediction_data = self.remove_counting_stats(prediction_data)
        prediction_data = self.format_column_names(prediction_data)
        
        return prediction_data


def example_usage():
    """Example showing how to use the composition-based classes"""
    # Example with batters
    print("\n" + "="*50)
    print("BATTER DATA PROCESSING EXAMPLE")
    print("="*50)
    
    # Create the base data processor for batters
    batter_processor = BatterDataProcessing(start_year=2019, end_year=2024)
    batter_processor.filter_data()
    batter_processor.calc_fantasy_points()
    
    # Create training processor using composition
    batter_training = TrainingDataProcessor(batter_processor)
    batter_training_data = batter_training.prepare_data()
    with (open('batter_training_data.txt', 'w')) as f:
        f.write(str(batter_training_data))
    print(f"Prepared {len(batter_training_data)} batter training datasets")
    
    # Create prediction processor using composition
    batter_prediction = Predict2025Processor(batter_processor)
    batter_prediction_data = batter_prediction.prepare_data()
    with (open('batter_prediction_data.txt', 'w')) as f:
        f.write(str(batter_prediction_data))
    print(f"Prepared batter prediction dataset for {batter_prediction_data['prediction_year']}")
    
    # Example with starters
    print("\n" + "="*50)
    print("STARTING PITCHER DATA PROCESSING EXAMPLE")
    print("="*50)
    
    # Create the base data processor for starters
    starter_processor = StarterDataProcessing(start_year=2019, end_year=2024)
    starter_processor.filter_data()
    starter_processor.calc_fantasy_points()
    
    # Create training processor using composition
    starter_training = TrainingDataProcessor(starter_processor)
    starter_training_data = starter_training.prepare_data()
    with (open('starter_training_data.txt', 'w')) as f:
        f.write(str(starter_training_data))

    print(f"Prepared {len(starter_training_data)} starter training datasets")
    
    # Create prediction processor using composition
    starter_prediction = Predict2025Processor(starter_processor)
    starter_prediction_data = starter_prediction.prepare_data()
    with (open('starter_prediction_data.txt', 'w')) as f:
        f.write(str(starter_prediction_data))
    print(f"Prepared starter prediction dataset for {starter_prediction_data['prediction_year']}")

    # Example with relievers
    print("\n" + "="*50)
    print("RELIEF PITCHER DATA PROCESSING EXAMPLE")
    print("="*50)
    
    # Create the base data processor for relievers
    reliever_processor = RelieverDataProcessing(start_year=2019, end_year=2024)
    reliever_processor.filter_data()
    reliever_processor.calc_fantasy_points()
    
    # Create training processor using composition
    reliever_training = TrainingDataProcessor(reliever_processor)
    reliever_training_data = reliever_training.prepare_data()
    with (open('reliever_training_data.txt', 'w')) as f:
        f.write(str(reliever_training_data))

    print(f"Prepared {len(reliever_training_data)} reliever training datasets")
    
    # Create prediction processor using composition
    reliever_prediction = Predict2025Processor(reliever_processor)
    reliever_prediction_data = reliever_prediction.prepare_data()
    with (open('reliever_prediction_data.txt', 'w')) as f:
        f.write(str(reliever_prediction_data))

    print(f"Prepared reliever prediction dataset for {reliever_prediction_data['prediction_year']}")


if __name__ == "__main__":
    example_usage()
