from ..DataProcessing.DataProcessing import DataProcessing, LeagueType
from ..FangraphsScraper.fangraphsScraper import PositionCategory
import pandas as pd
from typing import List, Dict, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class BatterDataProcessing(DataProcessing):
    """Concrete implementation for batter data processing"""
    
    def __init__(self, league_type: LeagueType, start_year: int = 2019, end_year: int = 2024) -> None:
        """Initialize the batter data processor.
        
        Args:
            league_type: Type of league for processing
            start_year: Starting year for data collection
            end_year: Ending year for data collection
        """
        super().__init__(PositionCategory.BATTER, league_type, start_year, end_year)
        self.stat_categories = ['R', 'RBI', 'HR', 'SO', 'TB', 'OPS', 'NSB']

    def _calc_net_stolen_bases(self) -> None:
        """Calculate net stolen bases (SB - CS) and add to dataframe."""
  
        sb = self.data['SB']
        cs = self.data['CS']
        nsb = sb - cs
        self.data['NSB'] = nsb

    def _calc_tb(self) -> None:
        """Calculate total bases and add to dataframe.
        
        Total Bases = 1B + (2 * 2B) + (3 * 3B) + (4 * HR)
        """
        singles = self.data['1B']
        doubles = self.data['2B']
        triples = self.data['3B']
        home_runs = self.data['HR']
        tb = singles + (2 * doubles) + (3 * triples) + (4 * home_runs)

        self.data['TB'] = tb
    
    def _calc_plate_discipline_score(self) -> None:
        """Calculate plate discipline score using multiple PCA components.
        
        Creates a composite score from various plate discipline metrics using
        principal component analysis to reduce dimensionality.
        """

        plate_discipline_columns = [
            'O-Swing%', 'Z-Swing%', 'Swing%', 'O-Contact%', 
            'Z-Contact%', 'Contact%', 'Zone%', 'F-Strike%', 'SwStr%'
        ]
        
        available_columns = [col for col in plate_discipline_columns if col in self.data.columns]
        if len(available_columns) < 3:
            print(f"Not enough plate discipline metrics available ({len(available_columns)}). Skipping PlateDisciplineScore.")
            return
            
        cols_to_invert = ['O-Swing%', 'SwStr%']
        available_cols_to_invert = [col for col in cols_to_invert if col in available_columns]

        plate_disc_data = self.data[available_columns].copy()
        plate_disc_data = plate_disc_data.fillna(plate_disc_data.mean())
        
        scaler = StandardScaler()
        plate_disc_data = pd.DataFrame(
            scaler.fit_transform(plate_disc_data),
            columns=available_columns
        )
        

        for col in available_cols_to_invert:
            plate_disc_data[col] = plate_disc_data[col] * -1
        
        n_components = min(3, len(available_columns))
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(plate_disc_data)

        if n_components == 3:
            composite = 0.60 * X_pca[:, 0] + 0.30 * X_pca[:, 1] + 0.10 * X_pca[:, 2]
        elif n_components == 2:
            composite = 0.70 * X_pca[:, 0] + 0.30 * X_pca[:, 1]
        else:
            composite = X_pca[:, 0]
        
        composite_min, composite_max = composite.min(), composite.max()
        plate_disc_score = (composite - composite_min) / (composite_max - composite_min) * 100

        self.data['PlateDisciplineScore'] = plate_disc_score

    def filter_data(self) -> None:
        """Filter and reshape the data for batters.
        
        Calculates derived metrics (plate discipline score, net stolen bases, 
        total bases) and selects relevant columns for batter analysis.
        """
        self._calc_plate_discipline_score()
        self._calc_net_stolen_bases()
        self._calc_tb()
        
        desired_columns = [
            'PlayerName', 'Age', 'Year', 'G', 'AVG', 'OBP', 'SLG', 'wOBA', 'wRC+', 
            'H', 'HBP', '1B', '2B', '3B', 'HR', 'R', 'RBI', 'BB%', 'K%', 'ISO', 'SB', 'CS', 'NSB',
            'HR/FB', 'GB/FB', 'LD%', 'GB%', 'FB%', 'xwOBA', 'xAVG', 'xSLG', 'EV', 'LA',
            'Barrel%', 'HardHit%', 'PlateDisciplineScore', 'BaseRunning', 'BABIP', 'Pull%',
            'Cent%', 'Oppo%', 'BB/K', 'Offense',  'OPS', 'SO', 'TB', 'wBsR'
        ]
        
        available_columns = [col for col in desired_columns if col in self.data.columns]
        self.data = self.data[available_columns]

        self.reshape_data()

    def calc_fantasy_points(self) -> None:
        """Calculate fantasy points for batters.
        
        Uses standard fantasy scoring metrics for batters including
        singles, doubles, triples, home runs, runs, RBIs, stolen bases,
        and hit by pitch.
        """

        fantasy_points = {}
        
        for year in self.years:
            year_str = str(year)
            if all(f'{year_str}_{stat}' in self.data.columns for stat in ['1B', '2B', '3B', 'HR', 'R', 'RBI', 'SB', 'HBP']):
                fantasy_points[f'{year_str}_TotalPoints'] = (
                    self.data[f'{year_str}_1B'] * 2.6 +
                    self.data[f'{year_str}_2B'] * 5.2 +
                    self.data[f'{year_str}_3B'] * 7.8 +
                    self.data[f'{year_str}_HR'] * 10.4 +
                    self.data[f'{year_str}_R'] * 1.9 +
                    self.data[f'{year_str}_RBI'] * 1.9 +
                    self.data[f'{year_str}_SB'] * 4.2 +
                    self.data[f'{year_str}_HBP'] * 2.6
                )
        
        if fantasy_points:
            self.data = pd.concat([
                self.data,
                pd.DataFrame(fantasy_points)
            ], axis=1)
    
    # Need to integrate injury history/games played
    def normalize_counting_stats(self) -> None:
        print("Normalizing counting stats...")
        feature_cols = [col for col in self.data.columns if col != 'PlayerName']
        
        stats_to_normalize = ['H', 'HBP', '1B', '2B', '3B', 'HR', 'R', 'RBI', 'SB', 'CS', 'NSB', 'SO', 'TB']

        for player, player_data in self.data.groupby('PlayerName'):
            player_idx = player_data.index[0]
            
            for stat in feature_cols:
                year = stat[:4]
                base_stat = stat[5:]

                if base_stat in stats_to_normalize:
                    games = self.data.loc[player_idx, f'{year}_G']

                    if not pd.isna(games) and games > 0:
                        original_value = self.data.loc[player_idx, stat]
                        normalized_value = (original_value / games) * 150
                        self.data.loc[player_idx, stat] = normalized_value
    
    def get_counting_stats(self) -> List[str]:
        """Return a list of counting stats for batters.
        
        Returns:
            List of column names representing counting stats to be removed
            during data normalization.
        """
        return ['1B', '2B', '3B', 'HR', 'R', 'RBI', 'HBP', 'SB']
