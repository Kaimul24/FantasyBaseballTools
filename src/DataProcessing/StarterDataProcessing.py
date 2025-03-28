from ..DataProcessing.DataProcessing import DataProcessing, LeagueType
from ..FangraphsScraper.fangraphsScraper import PositionCategory
import pandas as pd
from typing import List, Dict, Optional

class StarterDataProcessing(DataProcessing):
    """Concrete implementation for starting pitcher data processing"""
    
    def __init__(self, league_type: LeagueType, start_year: int = 2019, end_year: int = 2024) -> None:
        """Initialize the starter data processor.
        
        Args:
            league_type: Type of league for processing
            start_year: Starting year for data collection
            end_year: Ending year for data collection
        """
        super().__init__(PositionCategory.SP, league_type, start_year, end_year)
        self.stat_categories = ['ERA', 'WHIP', 'QS']
    
    def filter_data(self) -> None:
        """Filter and reshape the data for starting pitchers.
        
        Selects relevant columns for starter analysis and reshapes data
        for year-by-year comparison.
        """
        desired_columns = [
            'PlayerName', 'Age', 'Year', 'IP', 'ERA', 'FIP', 'xFIP', 'SIERA', 
            'K/9', 'BB/9', 'HR/9', 'BABIP', 'LOB%', 'GB%', 'HR/FB', 'K%', 'BB%', 
            'HBP', 'xERA', 'SO', 'BB', 'HR', 'ER', 'W', 'L', 'H','Barrel%', 
            'HardHit%', 'EV', 'C+SwStr%', 'SwStr%', 'Soft%', 'Med%', 'Hard%',
            'K-BB%', 'WHIP', 'GB/FB', 'LD%', 'FB%', 'LA', 'WAR', 'QS',
        ]
        self.data = self.data[desired_columns]
        
        self.reshape_data()

    def calc_fantasy_points(self) -> None:
        """Calculate fantasy points for starting pitchers.
        
        Uses standard fantasy scoring metrics for starters including
        IP, SO, W, ER, H, BB, and HBP.
        """
        fantasy_points = {}
        
        for year in self.years:
            year_str = str(year)
            stats_to_check = ['W', 'SO', 'ER', 'BB', 'H', 'HBP', 'IP']
            
            if all(f'{year_str}_{stat}' in self.data.columns for stat in stats_to_check):
                fantasy_points[f'{year_str}_TotalPoints'] = (
                    self.data[f'{year_str}_IP'] * 3.35 +
                    self.data[f'{year_str}_SO'] * 3.35 +
                    self.data[f'{year_str}_W'] * 8.35 +
                    self.data[f'{year_str}_ER'] * -2.55 +
                    self.data[f'{year_str}_H'] * -0.85 +
                    self.data[f'{year_str}_BB'] * -0.85 +
                    self.data[f'{year_str}_HBP'] * -0.85
                )
        
        if fantasy_points:
            self.data = pd.concat([
                self.data,
                pd.DataFrame(fantasy_points)
            ], axis=1)
    
    def get_counting_stats(self) -> List[str]:
        """Return a list of counting stats for starters.
        
        Returns:
            List of column names representing counting stats to be removed
            during data normalization.
        """
        return ['W', 'SO', 'ER', 'BB', 'H', 'HBP', 'IP']