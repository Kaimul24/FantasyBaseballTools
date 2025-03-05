from FangraphsScraper.fangraphsScraper import FangraphsScraper, PositionCategory
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from typing import List, TypedDict

class DatasetSplit(TypedDict):
    train_years: List[int]
    val_year: int
    test_year: int
    train_data: pd.DataFrame
    val_data: pd.DataFrame
    test_data: pd.DataFrame

class WeightedDatasetSplit(DatasetSplit):
    weighted_data: pd.DataFrame

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
    def __init__(self, position_category: PositionCategory, start_year: int = 2019, end_year: int = 2024):
        self.position_category = position_category
        data = FangraphsScraper(position_category, start_year=start_year, end_year=end_year).get_data()
        self.data = data[data['Year'] != 2020]
        self.years = sorted(self.data['Year'].unique())

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
        for year in self.years:
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

    def create_rolling_datasets(self) -> List[DatasetSplit]:
        """Create rolling train/validation/test splits"""
        # Get all years from column names and sort them
        years = sorted(list(set(
            int(col.split('_')[0]) 
            for col in self.data.columns 
            if '_' in col and col.split('_')[0].isdigit()
        )))
        
        # Remove 2020 if present
        years = [year for year in years if year != 2020]
        print(years)
        
        # Create windows: each window needs 2 training years, 1 validation year, and 1 test year
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
        
        return datasets

    def concat_training_windows(self, datasets: List[DatasetSplit]) -> List[WeightedDatasetSplit]:
        """
        Process each training window separately applying weights to years.
        For 3 years: 50% recent year, 30% recent year - 1, 20% recent year - 2.
        Returns list of processed training windows.
        """
        processed_windows = []
        
        for dataset in datasets:
            train_data = dataset['train_data'].copy()
            train_years = sorted(dataset['train_years'])
            
            weighted_data = pd.DataFrame({'PlayerName': train_data['PlayerName'].unique()})
            
            year_prefix = f"{train_years[-1]}_"
            base_stats = [col.replace(year_prefix, '') for col in train_data.columns 
                         if year_prefix in col]
            
            for stat in base_stats:
                year3_col = f"{train_years[2]}_{stat}"  # Recent year
                year2_col = f"{train_years[1]}_{stat}"  # Recent year - 1
                year1_col = f"{train_years[0]}_{stat}"  # Recent year - 2
                
                stat_data = pd.DataFrame()
                for year_col in [year1_col, year2_col, year3_col]:
                    if year_col in train_data.columns:
                        year_stats = train_data[['PlayerName', year_col]].copy()
                        stat_data = pd.merge(stat_data, year_stats, on='PlayerName', how='outer') if not stat_data.empty else year_stats

                weighted_stat = pd.Series(0, index=stat_data.index)
                
                # Set weights: 60% recent year, 40% older year
                weights = {year3_col: 0.5, year2_col: 0.3, year1_col: 0.2}
                
                # Apply weights
                for year_col, weight in weights.items():
                    if year_col in stat_data.columns:
                        weighted_stat += stat_data[year_col].fillna(0) * weight
                    
                weighted_data[stat] = weighted_stat
            
            processed_windows.append({
                'train_years': train_years,
                'test_year': dataset['test_year'],
                'weighted_data': weighted_data,
                'train_data': train_data, 
                'test_data': dataset['test_data'],
            })
        
        return processed_windows

    def remove_stats(self, windows: List[WeightedDatasetSplit]) -> List[WeightedDatasetSplit]:
        '''Remove total points and counting stats from training data'''
        for window in windows:
            weighted_data = window['weighted_data']
            cols_to_drop = ['1B', '2B', '3B', 'HR', 'R', 'RBI', 
                            'HBP', 'SB']

            weighted_data = weighted_data.drop(cols_to_drop, axis=1)
            window['weighted_data'] = weighted_data

        return windows

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
                col: col.replace("weighted_", "") 
                for col in weighted_cols 
                if col.startswith("weighted_") and col != "PlayerName"
            }
            window['weighted_data'] = window['weighted_data'].rename(columns=weighted_rename)
        
        return windows

    def prepare_data_for_modeling(self) -> List[WeightedDatasetSplit]:
        """
        Prepare data for modeling by creating datasets, applying weights, and preprocessing.
        Returns a list of preprocessed training windows ready for model training.
        """
        # Create rolling datasets
        print("Creating datasets...\n")
        datasets = self.create_rolling_datasets()
        print(f"Number of datasets: {len(datasets)}\n")

        # Get processed training windows
        print("Processing training windows...\n")
        processed_windows = self.concat_training_windows(datasets)

        # Remove year prefixes
        print("Removing year prefixes...\n")
        processed_windows = self.remove_year_prefixes(processed_windows)

        # Remove counting stats from training data
        print("Removing counting stats from training data...\n")
        processed_windows = self.remove_stats(processed_windows)
        
        return processed_windows

if __name__ == '__main__':
    batters = DataProcessing(PositionCategory.BATTER)
    batters.filter_data()
    batters.calc_fantasy_points()

    # Save reshaped data to file
    with open("reshaped_batter_data.txt", "w") as f:
        f.write(batters.data.to_string())

