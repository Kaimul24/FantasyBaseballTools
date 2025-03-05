from DataProcessing.DataProcessing import DataProcessing
from FangraphsScraper.fangraphsScraper import PositionCategory
import pandas as pd
from typing import List

class RelieverDataProcessing(DataProcessing):
    """Concrete implementation for relief pitcher data processing"""
    
    def __init__(self, start_year: int = 2019, end_year: int = 2024):
        super().__init__(PositionCategory.RP, start_year, end_year)
    
    def filter_data(self):
        """Filter and reshape the data for relief pitchers"""
        
        columns = ['PlayerName', 'Age', 'Year', 'G', 'GS', 'IP', 'ERA', 'FIP', 'xFIP', 'SIERA', 
                   'K/9', 'BB/9', 'HR/9', 'BABIP', 'LOB%', 'GB%', 'HR/FB', 'K%', 'BB%', 'HBP',
                   'xERA', 'Stuff+', 'SO', 'BB', 'H', 'HR', 'ER', 'W', 'L', 'SV', 'HLD', 
                   'Barrel%', 'HardHit%', 'EV', 'CSW%', 'SwStr%', 'Soft%', 'Med%', 'Hard%']
        
        # Keep only columns that exist
        columns = [col for col in columns if col in self.data.columns]
        self.data = self.data[columns]
        
        self.reshape_data()

    def calc_fantasy_points(self):
        """Calculate fantasy points for relief pitchers"""
        # Create dictionary to store fantasy points for each year
        fantasy_points = {}
        
        for year in self.years:
            year_str = str(year)
            stats_to_check = ['SV', 'HLD', 'SO', 'ER', 'BB', 'H', 'HBP', 'IP']
            
            if all(f'{year_str}_{stat}' in self.data.columns for stat in stats_to_check):
                # Common fantasy scoring for pitchers (adjust as needed)
                fantasy_points[f'{year_str}_TotalPoints'] = (
                    self.data[f'{year_str}_IP'] * 3 +     # Points per out
                    self.data[f'{year_str}_SO'] * 3 +      # Points per strikeout
                    self.data[f'{year_str}_SV'] * 8 +      # Points per win
                    self.data[f'{year_str}_HLD'] * 6 +      # Points per hold
                    self.data[f'{year_str}_ER'] * -3 +    # Negative points for earned runs
                    self.data[f'{year_str}_H'] * -1.3 +   # Negative points for hits
                    self.data[f'{year_str}_BB'] * -1.3 +   # Negative points for walks
                    self.data[f'{year_str}_HBP'] * -1.3   # Negative points for hit by pitch
                )
        
        # Concatenate all fantasy points columns at once
        if fantasy_points:
            self.data = pd.concat([
                self.data,
                pd.DataFrame(fantasy_points)
            ], axis=1)
    
    def get_counting_stats(self) -> List[str]:
        """Return a list of counting stats to remove for relievers"""
        return ['SV', 'HLD', 'SO', 'ER', 'BB', 'H', 'HBP', 'IP']