from abc import ABC, abstractmethod
from ..FangraphsScraper.fangraphsScraper import FangraphsScraper, PositionCategory
import pandas as pd
import numpy as np
from enum import Enum
from sklearn.impute import KNNImputer
from typing import List, TypedDict

class DatasetSplit(TypedDict):
    train_years: List[int]
    test_year: int
    train_data: pd.DataFrame
    test_data: pd.DataFrame

class WeightedDatasetSplit(DatasetSplit):
    weighted_data: pd.DataFrame

class PredictionDataset(TypedDict):
    train_years: List[int]
    prediction_year: int
    train_data: pd.DataFrame

class DataProcessing(ABC):
    """
    Abstract Base Class for processing and calculating fantasy points for baseball players.
    
    This class handles common data processing functions and defines abstract methods
    that should be implemented by position-specific subclasses.
    """
    def __init__(self, position_category: PositionCategory, start_year: int = 2019, end_year: int = 2024):
        self.position_category = position_category
        data = FangraphsScraper(position_category, start_year=start_year, end_year=end_year).get_data()
        self.data = data[data['Year'] != 2020]
        self.years = sorted(self.data['Year'].unique())
    
    @abstractmethod
    def filter_data(self):
        """Filter and reshape the data based on position category"""
        pass

    @abstractmethod
    def calc_fantasy_points(self):
        """Calculate fantasy points for each year"""
        pass

    @abstractmethod
    def get_counting_stats(self) -> List[str]:
        """Return a list of counting stats to remove for this position category"""
        pass
    
    def reshape_data(self):
        """Reshape data so each player has one row with columns grouped by year"""
        # Get list of columns except PlayerName
        feature_cols = [col for col in self.data.columns if col != 'PlayerName']
            
        # Store all possible stats across years to identify which ones might be missing
        all_possible_stats = set()
        for year in self.years:
            # Get stats for this year
            year_data = self.data[self.data['Year'] == year]
            if not year_data.empty:
                year_stats = [col for col in year_data.columns if col not in ['PlayerName', 'Year']]
                all_possible_stats.update(year_stats)
        
        print(f"Total unique stats across all years: {len(all_possible_stats)}")
        
        # Create empty list to store reshaped data
        reshaped_data = []
        
        # Group by player
        for player, player_data in self.data.groupby('PlayerName'):
            # Initialize player dict with name
            player_dict = {'PlayerName': player}
            
            # For each year, add all stats
            for year in self.years:
                year_data = player_data[player_data['Year'] == year]
                if not year_data.empty:
                    row = year_data.iloc[0]
                    
                    # Add available stats with year prefix
                    for col in feature_cols:
                        if col != 'Year' and col in row:
                            player_dict[f"{year}_{col}"] = row[col]
                        
                    # Also create placeholders for stats that exist in other years but not this one
                    for stat in all_possible_stats:
                        if stat not in year_data.columns and stat != 'Year':
                            player_dict[f"{year}_{stat}"] = np.nan
                
                # If player has no data for this year, add NaN for all stats
                else:
                    for stat in all_possible_stats:
                        if stat != 'Year':
                            player_dict[f"{year}_{stat}"] = np.nan
            
            reshaped_data.append(player_dict)
        
        # Convert to DataFrame
        df = pd.DataFrame(reshaped_data)
        
        # Prepare data for KNN imputation
        # First, separate PlayerName column
        player_names = df['PlayerName']
        
        # Get numeric columns for imputation (exclude PlayerName)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        print(f"Missing values before imputation: {df[numeric_cols].isna().sum().sum()}")
        
        # Initialize KNNImputer
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        
        # Impute missing values
        imputed_data = imputer.fit_transform(df[numeric_cols])
        
        # Create new DataFrame with imputed values
        imputed_df = pd.DataFrame(imputed_data, columns=numeric_cols, index=df.index)
        
        print(f"Missing values after imputation: {imputed_df.isna().sum().sum()}")
        
        # Reattach PlayerName column
        imputed_df.insert(0, 'PlayerName', player_names)
        
        # Reorder columns to group by year
        year_groups = []
        for year in self.years:
            year_cols = [col for col in imputed_df.columns if str(year) in col]
            year_groups.extend(sorted(year_cols))
        
        # Final column order: PlayerName followed by year groups
        final_cols = ['PlayerName'] + year_groups
        self.data = imputed_df[final_cols]
        
        # Print summary of what was imputed
        stats_added = {}
        for year in self.years:
            year_str = str(year)
            year_cols = [col.replace(f"{year_str}_", "") for col in self.data.columns if col.startswith(f"{year_str}_")]
            stats_added[year] = len(year_cols)
            
        print("Stats per year after imputation:")
        for year, count in stats_added.items():
            print(f"  {year}: {count} stats")
            
        return self.data

    def remove_year_prefixes(self, windows: List[WeightedDatasetSplit]) -> List[WeightedDatasetSplit]:
        '''Remove year prefixes from validation and test data columns'''
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
            
            # Also remove 'weighted_' prefix from weighted_data
            weighted_cols = window['weighted_data'].columns
            weighted_rename = {
                col: col.replace("weighted_", "") for col in weighted_cols 
                if col.startswith("weighted_") and col != "PlayerName"
            }
            window['weighted_data'] = window['weighted_data'].rename(columns=weighted_rename)
        
        return windows
    
    def filter_and_calc_points(self):
        """Filter data and calculate fantasy points wrapper function"""
        self.filter_data()
        self.calc_fantasy_points()
        