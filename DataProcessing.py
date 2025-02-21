from FangraphsScraper.fangraphsScraper import FangraphsScraper, PositionCategory
import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer

class DataProcessing:
    """
    DataProcessing module for processing and calculating fantasy points for baseball players.
    Classes:
        DataProcessing: A class to process and calculate fantasy points for players based on their position category.
    Methods:
        __init__(self, position_category: PositionCategory):
            Initializes the DataProcessing class with the given position category and retrieves data using FangraphsScraper.
        filter_data(self):
            Filters the data based on the position category. Currently, it filters columns for batters.
        calc_fantasy_points(self):
            Calculates fantasy points for players based on their performance metrics and position category. Returns the updated dataframe with fantasy points.
    Usage:
        This module can be run as a script to process and calculate fantasy points for batters, and save the results to text files.
    """
    def __init__(self, position_category: PositionCategory):
        self.position_category = position_category
        # Fetch data in two ranges to exclude 2020
        data_2019 = FangraphsScraper(position_category, 2019, 2019).get_data()
        data_2021_2024 = FangraphsScraper(position_category, 2021, 2024).get_data()
        # Combine the data
        self.data = pd.concat([data_2019, data_2021_2024], ignore_index=True)

    def calc_plate_discipline_score(self):
        """Calculate plate discipline score using multiple PCA components."""
        plate_discipline_columns = [
            'O-Swing%', 'Z-Swing%', 'Swing%', 'O-Contact%', 
            'Z-Contact%', 'Contact%', 'Zone%', 'F-Strike%', 'SwStr%'
        ]
        cols_to_invert = ['O-Swing%', 'SwStr%']
        
        # Create a copy of the data for PCA
        plate_disc_data = self.data[plate_discipline_columns].copy()
        
        # Scale the data
        scaler = StandardScaler()
        plate_disc_data = pd.DataFrame(
            scaler.fit_transform(plate_disc_data),
            columns=plate_discipline_columns
        )
        
        # Invert 'lower is better' stats
        for col in cols_to_invert:
            plate_disc_data[col] = plate_disc_data[col] * -1
        
        # PCA with more components
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(plate_disc_data)
        
        # Calculate composite score
        composite = 0.60 * X_pca[:, 0] + 0.40 * X_pca[:, 1] + 0.1 * X_pca[:, 2]
        
        # Normalize to 0-100
        composite_min, composite_max = composite.min(), composite.max()
        plate_disc_score = (composite - composite_min) / (composite_max - composite_min) * 100
        
        # Concatenate new column with original dataframe
        self.data = pd.concat([
            self.data,
            pd.Series(plate_disc_score, name='PlateDisciplineScore', index=self.data.index)
        ], axis=1)

    ### MAY NOT NEED
    def transform_ages(self):
        """Transform age data with polynomial features."""
        # Calculate all age transformations at once
        age_transforms = pd.DataFrame({
            'log(Age-28)^2': self.data['Age'].apply(lambda x: np.log((abs(x - 28) + 1)**2)),
            #'Age^3': self.data['Age'].apply(lambda x: x**3)
        }, index=self.data.index)
        
        # Concatenate new columns and drop original Age column
        self.data = pd.concat([
            self.data.drop('Age', axis=1),
            age_transforms
        ], axis=1)
    
    def reshape_data(self):
        """Reshape data so each player has one row with columns grouped by year"""
        # Get list of columns except PlayerName
        feature_cols = [col for col in self.data.columns if col != 'PlayerName']
        years = sorted(self.data['Year'].unique())
        
        # Create empty list to store reshaped data
        reshaped_data = []
        
        # Group by player
        for player, player_data in self.data.groupby('PlayerName'):
            # Initialize player dict with name
            player_dict = {'PlayerName': player}
            
            # For each year, add all stats
            for year in years:
                year_data = player_data[player_data['Year'] == year]
                if not year_data.empty:
                    row = year_data.iloc[0]
                    # Add each stat with year prefix, maintaining year grouping
                    year_stats = {
                        f"{year}_stats": {
                            col: row[col] for col in feature_cols if col != 'Year'
                        }
                    }
                    # Flatten the year_stats dictionary with year prefix
                    for col, value in year_stats[f"{year}_stats"].items():
                        player_dict[f"{year}_{col}"] = value
            
            reshaped_data.append(player_dict)
        
        # Convert to DataFrame
        df = pd.DataFrame(reshaped_data)
        
        # Prepare data for KNN imputation
        # First, separate PlayerName column
        player_names = df['PlayerName']
        
        # Get numeric columns for imputation (exclude PlayerName)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Initialize and fit KNNImputer
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        
        # Impute missing values
        imputed_data = imputer.fit_transform(df[numeric_cols])
        
        # Create new DataFrame with imputed values
        imputed_df = pd.DataFrame(imputed_data, columns=numeric_cols, index=df.index)
        
        # Reattach PlayerName column
        imputed_df.insert(0, 'PlayerName', player_names)
        
        # Reorder columns to group by year
        year_groups = []
        for year in years:
            year_cols = [col for col in imputed_df.columns if str(year) in col]
            year_groups.extend(sorted(year_cols))
        
        # Final column order: PlayerName followed by year groups
        final_cols = ['PlayerName'] + year_groups
        self.data = imputed_df[final_cols]
        return self.data

    def filter_data(self):
        """Filter and reshape the data based on position category"""
        if self.position_category == PositionCategory.BATTER:
            self.calc_plate_discipline_score()
            
            columns = ['PlayerName', 'Age', 'Year', 'G', 'PA', 'AB', 'AVG', 'OBP', 'SLG', 'wOBA', 'wRC+', 'H', 'HBP',
                      '1B', '2B', '3B', 'HR', 'R', 'RBI', 'BB%', 'K%', 'ISO', 'SB', 'CS', 'HR/FB', 'GB/FB', 'LD%', 'GB%', 'FB%',
                      'xwOBA', 'xAVG', 'xSLG', 'EV', 'LA', 'Barrel%', 'HardHit%', 'PlateDisciplineScore', 'BaseRunning',
                      'BABIP', 'Pull%', 'Cent%', 'Oppo%', 'BB/K', 'Offense']
            
            self.data = self.data[columns]
            
            # Reshape data after filtering columns
            self.reshape_data()

    def calc_fantasy_points(self):
        """Calculate fantasy points for each year"""
        years = set([int(col.split('_')[0]) for col in self.data.columns if '_' in col])
        
        # Create dictionary to store fantasy points for each year
        fantasy_points = {}
        
        for year in years:
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

if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    batters = DataProcessing(PositionCategory.BATTER)
    batters.filter_data()
    batters.calc_fantasy_points()

    # Save reshaped data to file
    with open("reshaped_batter_data.txt", "w") as f:
        print(batters.data, file=f)
