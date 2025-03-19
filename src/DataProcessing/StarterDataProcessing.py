from ..DataProcessing.DataProcessing import DataProcessing
from ..FangraphsScraper.fangraphsScraper import PositionCategory
import pandas as pd
import numpy as np
from typing import List

class StarterDataProcessing(DataProcessing):
    """Concrete implementation for starting pitcher data processing"""
    
    def __init__(self, start_year: int = 2019, end_year: int = 2024):
        super().__init__(PositionCategory.SP, start_year, end_year)
    
    def filter_data(self):
        """Filter and reshape the data for starting pitchers"""
        # Define columns we want to keep if they exist
        desired_columns = [
            'PlayerName', 'Age', 'Year', 'IP', 'ERA', 'FIP', 'xFIP', 'SIERA', 
            'K/9', 'BB/9', 'HR/9', 'BABIP', 'LOB%', 'GB%', 'HR/FB', 'K%', 'BB%', 
            'HBP', 'xERA', 'SO', 'BB', 'HR', 'ER', 'W', 'L', 'H','Barrel%', 
            'HardHit%', 'EV', 'C+SwStr%', 'SwStr%', 'Soft%', 'Med%', 'Hard%',
            'K-BB%', 'WHIP', 'GB/FB', 'LD%', 'FB%', 'LA', 'WAR', 'QS'
        ]
        self.data = self.data[desired_columns]
        
        self.reshape_data()

    def calc_fantasy_points(self):
        """Calculate fantasy points for starting pitchers"""
        # Create dictionary to store fantasy points for each year
        fantasy_points = {}
        
        for year in self.years:
            year_str = str(year)
            stats_to_check = ['W', 'SO', 'ER', 'BB', 'H', 'HBP', 'IP']
            
            if all(f'{year_str}_{stat}' in self.data.columns for stat in stats_to_check):
                # Common fantasy scoring for pitchers (adjust as needed)
                fantasy_points[f'{year_str}_TotalPoints'] = (
                    self.data[f'{year_str}_IP'] * 3.35 +     # Points per out
                    self.data[f'{year_str}_SO'] * 3.35 +      # Points per strikeout
                    self.data[f'{year_str}_W'] * 8.35 +      # Points per win
                    self.data[f'{year_str}_ER'] * -2.55 +    # Negative points for earned runs
                    self.data[f'{year_str}_H'] * -0.85 +   # Negative points for hits
                    self.data[f'{year_str}_BB'] * -0.85 +   # Negative points for walks
                    self.data[f'{year_str}_HBP'] * -0.85   # Negative points for hit by pitch
                )
        
        # Concatenate all fantasy points columns at once
        if fantasy_points:
            self.data = pd.concat([
                self.data,
                pd.DataFrame(fantasy_points)
            ], axis=1)
    
    def get_counting_stats(self) -> List[str]:
        """Return a list of counting stats to remove for starters"""
        return ['W', 'SO', 'ER', 'BB', 'H', 'HBP', 'IP']