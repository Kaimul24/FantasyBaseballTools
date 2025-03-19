import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..DataProcessing.DataProcessing import DataProcessing
from ..FangraphsScraper.fangraphsScraper import PositionCategory
import pandas as pd
from typing import List
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


class BatterDataProcessing(DataProcessing):
    """Concrete implementation for batter data processing"""
    
    def __init__(self, start_year: int = 2019, end_year: int = 2024):
        super().__init__(PositionCategory.BATTER, start_year, end_year)
        self.stat_categories = ['R', 'RBI', 'HR', 'SO', 'TB', 'OPS', 'NSB']

    def calc_net_stolen_bases(self):
        sb = self.data['SB']
        cs = self.data['CS']
        nsb = sb - cs
        self.data = pd.concat([
            self.data,
            pd.Series(nsb, name='NSB', index=self.data.index)
        ], axis=1)

    def calc_tb(self):
        singles = self.data['1B']
        doubles = self.data['2B']
        triples = self.data['3B']
        home_runs = self.data['HR']
        tb = singles + (2 * doubles) + (3 * triples) + (4 * home_runs)

        self.data = pd.concat([
            self.data,
            pd.Series(tb, name='TB', index=self.data.index)
        ], axis=1)
    
    def calc_plate_discipline_score(self):
        """Calculate plate discipline score using multiple PCA components."""
        plate_discipline_columns = [
            'O-Swing%', 'Z-Swing%', 'Swing%', 'O-Contact%', 
            'Z-Contact%', 'Contact%', 'Zone%', 'F-Strike%', 'SwStr%'
        ]
        
        # Check which plate discipline columns are available
        available_columns = [col for col in plate_discipline_columns if col in self.data.columns]
        if len(available_columns) < 3:
            print(f"Not enough plate discipline metrics available ({len(available_columns)}). Skipping PlateDisciplineScore.")
            return
            
        cols_to_invert = ['O-Swing%', 'SwStr%']
        available_cols_to_invert = [col for col in cols_to_invert if col in available_columns]
        
        # Create a copy of the data for PCA with available columns
        plate_disc_data = self.data[available_columns].copy()
        
        # Handle missing values for PCA
        plate_disc_data = plate_disc_data.fillna(plate_disc_data.mean())
        
        # Scale the data
        scaler = StandardScaler()
        plate_disc_data = pd.DataFrame(
            scaler.fit_transform(plate_disc_data),
            columns=available_columns
        )
        
        # Invert 'lower is better' stats that are available
        for col in available_cols_to_invert:
            plate_disc_data[col] = plate_disc_data[col] * -1
        
        # PCA with appropriate number of components
        n_components = min(3, len(available_columns))
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(plate_disc_data)
        
        # Calculate composite score with weights
        if n_components == 3:
            composite = 0.60 * X_pca[:, 0] + 0.30 * X_pca[:, 1] + 0.10 * X_pca[:, 2]
        elif n_components == 2:
            composite = 0.70 * X_pca[:, 0] + 0.30 * X_pca[:, 1]
        else:
            composite = X_pca[:, 0]
        
        # Normalize to 0-100
        composite_min, composite_max = composite.min(), composite.max()
        plate_disc_score = (composite - composite_min) / (composite_max - composite_min) * 100
        
        # Concatenate new column with original dataframe
        self.data = pd.concat([
            self.data,
            pd.Series(plate_disc_score, name='PlateDisciplineScore', index=self.data.index)
        ], axis=1)

    def filter_data(self):
        """Filter and reshape the data for batters"""
        self.calc_plate_discipline_score()
        self.calc_net_stolen_bases()
        self.calc_tb()

        # Define columns we want to keep if they exist
        desired_columns = [
            'PlayerName', 'Age', 'Year', 'G', 'AVG', 'OBP', 'SLG', 'wOBA', 'wRC+', 
            'H', 'HBP', '1B', '2B', '3B', 'HR', 'R', 'RBI', 'BB%', 'K%', 'ISO', 'SB', 'CS', 'NSB',
            'HR/FB', 'GB/FB', 'LD%', 'GB%', 'FB%', 'xwOBA', 'xAVG', 'xSLG', 'EV', 'LA',
            'Barrel%', 'HardHit%', 'PlateDisciplineScore', 'BaseRunning', 'BABIP', 'Pull%',
            'Cent%', 'Oppo%', 'BB/K', 'Offense', 'WAR', 'OPS', 'SO', 'TB', 'wBsR'
        ]
        
        # Keep only columns that exist in the data
        available_columns = [col for col in desired_columns if col in self.data.columns]
        self.data = self.data[available_columns]
        assert 'wBsR' in self.data.columns

        self.reshape_data()

    def calc_fantasy_points(self):
        """Calculate fantasy points for batters"""
        # Create dictionary to store fantasy points for each year
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
        
        # Concatenate all fantasy points columns at once
        if fantasy_points:
            self.data = pd.concat([
                self.data,
                pd.DataFrame(fantasy_points)
            ], axis=1)
    
    def get_counting_stats(self) -> List[str]:
        """Return a list of counting stats to remove for batters"""
        return ['1B', '2B', '3B', 'HR', 'R', 'RBI', 'HBP', 'SB']
    