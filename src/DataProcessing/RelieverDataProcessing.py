from ..DataProcessing.DataProcessing import DataProcessing, LeagueType
from ..FangraphsScraper.fangraphsScraper import PositionCategory
import pandas as pd
from typing import List, Dict, Optional

class RelieverDataProcessing(DataProcessing):
    """Concrete implementation for relief pitcher data processing"""
    
    def __init__(self, league_type: LeagueType, start_year: int = 2019, end_year: int = 2024) -> None:
        """Initialize the reliever data processor.
        
        Args:
            league_type: Type of league for processing
            start_year: Starting year for data collection
            end_year: Ending year for data collection
        """
        super().__init__(PositionCategory.RP, league_type, start_year, end_year)
        self.stat_categories = ['SO', 'ERA', 'WHIP', 'NSVH']
    
    def filter_data(self) -> None:
        """Filter and reshape the data for relief pitchers.
        
        Calculates NSVH stat and selects relevant columns for reliever analysis.
        """
        self._calc_nsvh()
        
        desired_columns = [
            'PlayerName', 'Age', 'Year', 'IP', 'ERA', 'FIP', 'xFIP', 'SIERA', 
            'K/9', 'BB/9', 'HR/9', 'BABIP', 'LOB%', 'GB%', 'HR/FB', 'K%', 'BB%', 
            'HBP', 'xERA', 'SO', 'BB', 'H', 'HR', 'ER', 'SV', 'HLD', 'Barrel%', 
            'HardHit%', 'EV', 'C+SwStr%', 'SwStr%', 'Soft%', 'Med%', 'Hard%',
            'K-BB%', 'WHIP', 'GB/FB', 'LD%', 'FB%', 'LA', 'WAR', 'BS', 'NSVH', 'G'
        ]
        
        self.data = self.data[desired_columns]

        self.reshape_data()

    def _calc_nsvh(self) -> None:
        """Calculate NSVH (Saves - Holds - Blown Saves) and add to dataframe."""
        sv = self.data["SV"]
        hld = self.data["HLD"]
        bs = self.data["BS"]

        nsvh = sv - hld - bs
        self.data['NSVH'] = nsvh

    def calc_fantasy_points(self) -> None:
        """Calculate fantasy points for relief pitchers.
        
        Uses standard fantasy scoring metrics for relievers including
        IP, SO, SV, HLD, ER, H, BB, and HBP.
        """
        
        fantasy_points = {}
        
        for year in self.years:
            year_str = str(year)
            stats_to_check = ['SV', 'HLD', 'SO', 'ER', 'BB', 'H', 'HBP', 'IP']
            
            if all(f'{year_str}_{stat}' in self.data.columns for stat in stats_to_check):
                fantasy_points[f'{year_str}_TotalPoints'] = (
                    self.data[f'{year_str}_IP'] * 3.4 +
                    self.data[f'{year_str}_SO'] * 3.4 +
                    self.data[f'{year_str}_SV'] * 8.4 +
                    self.data[f'{year_str}_HLD'] * 6.4 +
                    self.data[f'{year_str}_ER'] * -2.6 +
                    self.data[f'{year_str}_H'] * -0.9 +
                    self.data[f'{year_str}_BB'] * -0.9 +
                    self.data[f'{year_str}_HBP'] * -0.9
                )

        if fantasy_points:
            self.data = pd.concat([
                self.data,
                pd.DataFrame(fantasy_points)
            ], axis=1)

    def normalize_counting_stats(self) -> None:
        print("Normalizing counting stats...")
        feature_cols = [col for col in self.data.columns if col != 'PlayerName']
        
        stats_to_normalize = ['IP', 'HBP', 'SO', 'BB', 'HR', 'ER', 'H', 'WAR', 'SV', 'HLD', 'BS', 'NSVH']

        for player, player_data in self.data.groupby('PlayerName'):
            player_idx = player_data.index[0]
            
            for stat in feature_cols:
                year = stat[:4]
                base_stat = stat[5:]

                if base_stat in stats_to_normalize:
                    innings_pitched = self.data.loc[player_idx, f'{year}_IP']

                    if not pd.isna(innings_pitched) and innings_pitched > 0:
                        original_value = self.data.loc[player_idx, stat]
                        normalized_value = (original_value / innings_pitched) * 65
                        self.data.loc[player_idx, stat] = normalized_value
    
    def get_counting_stats(self) -> List[str]:
        """Return a list of counting stats for relievers.
        
        Returns:
            List of column names representing counting stats to be removed
            during data normalization.
        """
        return ['SV', 'HLD', 'SO', 'ER', 'BB', 'H', 'HBP', 'IP']