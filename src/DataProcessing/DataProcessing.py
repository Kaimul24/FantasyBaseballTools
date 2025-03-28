from abc import ABC, abstractmethod
from ..FangraphsScraper.fangraphsScraper import FangraphsScraper, PositionCategory
import pandas as pd
import numpy as np
from enum import Enum
from sklearn.impute import KNNImputer
from typing import List, TypedDict, Dict, Set, Optional, Union, Any

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

class LeagueType(Enum):
    POINTS = 1
    CATEGORIES = 2

class DataProcessing(ABC):
    """
    Abstract Base Class for processing and calculating fantasy points for baseball players.
    
    This class handles common data processing functions and defines abstract methods
    that should be implemented by position-specific subclasses.
    """
    def __init__(self, position_category: PositionCategory, league_type: LeagueType = LeagueType.POINTS, start_year: int = 2019, end_year: int = 2024) -> None:
        """
        Initialize the DataProcessing object.
        
        Parameters:
            position_category: The category of player positions to process
            league_type: Type of fantasy league (points or categories)
            start_year: The earliest year of data to retrieve
            end_year: The latest year of data to retrieve
        """
        self.position_category = position_category
        data = FangraphsScraper(position_category, start_year=start_year, end_year=end_year).get_data()
        self.data = data[data['Year'] != 2020]
        self.years = sorted(self.data['Year'].unique())
        self.league_type = league_type
    
    @abstractmethod
    def filter_data(self) -> pd.DataFrame:
        """
        Filter and reshape the data based on position category.
        
        This method should be implemented by subclasses to apply position-specific
        filtering logic to the dataset.
        
        Returns:
            Filtered pandas DataFrame
        """
        pass

    @abstractmethod
    def calc_fantasy_points(self) -> pd.DataFrame:
        """
        Calculate fantasy points for each year.
        
        This method should be implemented by subclasses to calculate fantasy points
        based on position-specific scoring rules.
        
        Returns:
            DataFrame with fantasy points calculated for each player
        """
        pass

    @abstractmethod
    def get_counting_stats(self) -> List[str]:
        """
        Return a list of counting stats to remove for this position category.
        
        This method should be implemented by subclasses to identify which
        statistical categories should be excluded from analysis.
        
        Returns:
            List of column names representing counting stats
        """
        pass
    
    def reshape_data(self) -> pd.DataFrame:
        """
        Reshape data so each player has one row with columns grouped by year.
        
        This method transforms the data from a format where each player has multiple rows
        (one per year) to a format where each player has a single row with stats for each
        year as separate columns. Missing values are imputed using KNN.
        
        Returns:
            Reshaped and imputed DataFrame
        """
        feature_cols = [col for col in self.data.columns if col != 'PlayerName']
            
        all_possible_stats = set()
        for year in self.years:
            year_data = self.data[self.data['Year'] == year]
            if not year_data.empty:
                year_stats = [col for col in year_data.columns if col not in ['PlayerName', 'Year']]
                all_possible_stats.update(year_stats)
        
        print(f"Total unique stats across all years: {len(all_possible_stats)}")
        
        reshaped_data = []
        
        for player, player_data in self.data.groupby('PlayerName'):
            player_dict = {'PlayerName': player}
            
            for year in self.years:
                year_data = player_data[player_data['Year'] == year]
                if not year_data.empty:
                    row = year_data.iloc[0]
                    
                    for col in feature_cols:
                        if col != 'Year' and col in row:
                            player_dict[f"{year}_{col}"] = row[col]
                        
                    for stat in all_possible_stats:
                        if stat not in year_data.columns and stat != 'Year':
                            player_dict[f"{year}_{stat}"] = np.nan
                
                else:
                    for stat in all_possible_stats:
                        if stat != 'Year':
                            player_dict[f"{year}_{stat}"] = np.nan
            
            reshaped_data.append(player_dict)
        
        df = pd.DataFrame(reshaped_data)
        
        player_names = df['PlayerName']
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        print(f"Missing values before imputation: {df[numeric_cols].isna().sum().sum()}")
        
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        
        imputed_data = imputer.fit_transform(df[numeric_cols])
        
        imputed_df = pd.DataFrame(imputed_data, columns=numeric_cols, index=df.index)
        
        print(f"Missing values after imputation: {imputed_df.isna().sum().sum()}")
        
        imputed_df.insert(0, 'PlayerName', player_names)
        
        year_groups = []
        for year in self.years:
            year_cols = [col for col in imputed_df.columns if str(year) in col]
            year_groups.extend(sorted(year_cols))
        
        final_cols = ['PlayerName'] + year_groups
        self.data = imputed_df[final_cols]
        
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
        """
        Remove year prefixes from validation and test data columns.
        
        This function transforms column names by removing the year prefix from test data columns
        and the 'weighted_' prefix from weighted data columns.
        
        Parameters:
            windows: List of dataset splits with train, test, and weighted data
            
        Returns:
            The same list of windows with renamed columns in test_data and weighted_data
        """
        for window in windows:
            
            test_year = str(window['test_year'])
            test_cols = window['test_data'].columns
            test_rename = {
                col: col.replace(f"{test_year}_", "") 
                for col in test_cols 
                if col.startswith(f"{test_year}_")
            }
            window['test_data'] = window['test_data'].rename(columns=test_rename)
            
            weighted_cols = window['weighted_data'].columns
            weighted_rename = {
                col: col.replace("weighted_", "") for col in weighted_cols 
                if col.startswith("weighted_") and col != "PlayerName"
            }
            window['weighted_data'] = window['weighted_data'].rename(columns=weighted_rename)
        
        return windows
    
    def filter_and_calc_points(self) -> pd.DataFrame:
        """
        Filter data and calculate fantasy points wrapper function.
        
        This is a convenience method that calls filter_data followed by calc_fantasy_points
        to perform the complete data processing pipeline.
        
        Returns:
            DataFrame with filtered data and calculated fantasy points
        """
        self.filter_data()
        self.calc_fantasy_points()
        