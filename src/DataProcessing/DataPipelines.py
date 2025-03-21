import pandas as pd
from typing import List, Union, Dict, Any
from enum import Enum
from abc import ABC, abstractmethod
from ..DataProcessing.DataProcessing import (
    DatasetSplit, WeightedDatasetSplit, PredictionDataset, DataProcessing
)

class LeagueType(Enum):
    POINTS = 1
    CATEGORIES = 2

class BaseDataPrep(ABC):
    def __init__(self, data_processor: DataProcessing, league_type: LeagueType = LeagueType.POINTS):
        self.data_processor = data_processor
        self.data = data_processor.data
        self.league_type = league_type
    
    @abstractmethod
    def create_dataset(self):
        """Create dataset splits specific to the preparation type"""
        pass
    
    def apply_weights(self, dataset_data: Union[List[DatasetSplit], PredictionDataset]) -> Union[List[WeightedDatasetSplit], PredictionDataset]:
        """
        Apply weights to the datasets based on years.
        For 3 years: 50% recent year, 30% recent year - 1, 20% recent year - 2.
        Works for both training and prediction datasets.
        """
        print("Applying weights to data...\n")
        
        if isinstance(dataset_data, dict):  # Single prediction dataset
            return self._apply_weights_to_single(dataset_data)
        else:  # List of training datasets
            return [self._apply_weights_to_single(dataset) for dataset in dataset_data]
    
    def _apply_weights_to_single(self, dataset: Union[DatasetSplit, PredictionDataset]) -> Union[WeightedDatasetSplit, PredictionDataset]:
        """Helper method to apply weights to a single dataset"""
        train_data = dataset['train_data'].copy()
        train_years = sorted(dataset.get('train_years', []))
        
        # Initialize with just PlayerName column
        weighted_data_base = pd.DataFrame({'PlayerName': train_data['PlayerName'].unique()})
        
        year_prefix = f"{train_years[-1]}_"
        base_stats = [col.replace(year_prefix, '') for col in train_data.columns 
                     if year_prefix in col]
        
        # Create dictionary to store weighted stats
        weighted_stats = {}
        
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
            if year3_col: weights[year3_col] = 0.4
            if year2_col: weights[year2_col] = 0.35
            if year1_col: weights[year1_col] = 0.25
            
            # Normalize weights if some years are missing
            if weights:
                total_weight = sum(weights.values())
                weights = {k: v/total_weight for k, v in weights.items()}
            
            # Apply weights
            for year_col, weight in weights.items():
                if year_col in stat_data.columns:
                    weighted_stat += stat_data[year_col].fillna(0) * weight
            
            # Store weighted stat in dictionary
            weighted_stats[stat] = weighted_stat
            
        # Create DataFrame from all weighted stats at once
        weighted_stats_df = pd.DataFrame(weighted_stats, index=weighted_data_base.index)
        
        # Combine with the base DataFrame containing PlayerName
        weighted_data = pd.concat([weighted_data_base, weighted_stats_df], axis=1)
        
        result = dataset.copy()
        result['weighted_data'] = weighted_data
        return result
    
    def remove_counting_stats(self, dataset_data: Union[List[WeightedDatasetSplit], PredictionDataset]) -> Union[List[WeightedDatasetSplit], PredictionDataset]:
        """Remove counting stats from weighted data"""
        print("Removing counting stats from data...\n")
        
        # Get counting stats from the underlying data processor
        cols_to_drop = self.data_processor.get_counting_stats()
        
        if isinstance(dataset_data, dict):  # Single prediction dataset
            return self._remove_counting_stats_single(dataset_data, cols_to_drop)
        else:  # List of training datasets
            return [self._remove_counting_stats_single(window, cols_to_drop) for window in dataset_data]
    
    def _remove_counting_stats_single(self, dataset: Union[WeightedDatasetSplit, PredictionDataset], cols_to_drop: List[str]) -> Union[WeightedDatasetSplit, PredictionDataset]:
        """Helper method to remove counting stats from a single dataset"""
        result = dataset.copy()
        weighted_data = result['weighted_data']
        
        # Drop only columns that actually exist
        cols_to_remove = [col for col in cols_to_drop if col in weighted_data.columns]
        if cols_to_remove:
            weighted_data = weighted_data.drop(cols_to_remove, axis=1)
        
        result['weighted_data'] = weighted_data
        return result
    
    def format_column_names(self, dataset_data: Union[List[WeightedDatasetSplit], PredictionDataset]) -> Union[List[WeightedDatasetSplit], PredictionDataset]:
        """Format column names by removing prefixes"""
        print("Formatting column names...\n")
        
        if isinstance(dataset_data, dict):  # Single prediction dataset
            return self._format_column_names_single(dataset_data)
        else:  # List of training datasets
            return [self._format_column_names_single(window) for window in dataset_data]
    
    def _format_column_names_single(self, dataset: Union[WeightedDatasetSplit, PredictionDataset]) -> Union[WeightedDatasetSplit, PredictionDataset]:
        """Helper method to format column names in a single dataset"""
        result = dataset.copy()
        
        # Handle test data if it exists (only in training datasets)
        if 'test_data' in result and 'test_year' in result:
            test_year = str(result['test_year'])
            test_cols = result['test_data'].columns
            test_rename = {
                col: col.replace(f"{test_year}_", "") 
                for col in test_cols 
                if col.startswith(f"{test_year}_")
            }
            result['test_data'] = result['test_data'].rename(columns=test_rename)
        
        # Also remove 'weighted_' prefix from weighted_data if present
        if 'weighted_data' in result:
            weighted_cols = result['weighted_data'].columns
            weighted_rename = {
                col: col.replace("weighted_", "") 
                for col in weighted_cols 
                if col.startswith("weighted_") and col != "PlayerName"
            }
            result['weighted_data'] = result['weighted_data'].rename(columns=weighted_rename)
        
        return result
    
    def prepare_data(self):
        """
        Define the data preparation workflow. Child classes should customize this if needed.
        """
        dataset = self.create_dataset()
        dataset = self.apply_weights(dataset)
        
        if self.league_type == LeagueType.POINTS:
            dataset = self.remove_counting_stats(dataset)
        
        dataset = self.format_column_names(dataset)
        return dataset
    
class TrainingDataPrep(BaseDataPrep):
    """
    Class for processing data for model training.
    Includes function for creating rolling datasets.
    """
        
    def create_dataset(self) -> List[DatasetSplit]:
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

class PredictionDataPrep(BaseDataPrep):
    """
    Class for processing data to predict future fantasy points.
    Uses the most recent 3 years data to make predictions.
    """

    def create_dataset(self) -> PredictionDataset:
        """Create a single prediction dataset using recent years"""
        print("Creating prediction dataset...\n")
        
        # Get years from data
        years = sorted([int(col.split('_')[0]) 
                       for col in self.data.columns 
                       if '_' in col and col.split('_')[0].isdigit()])
        
        # Take the 3 most recent years
        recent_years = sorted(set(years))
        
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
