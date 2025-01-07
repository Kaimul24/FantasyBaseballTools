from FangraphsScraper.fangraphsScraper import FangraphsScraper, PositionCategory
import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
        self.data = FangraphsScraper(position_category, 2021, 2024).get_data()

    def calc_plate_discipline_score(self):
        """Calculate plate discipline score using multiple PCA components."""
        plate_discipline_columns = [
            'O-Swing%', 'Z-Swing%', 'Swing%', 'O-Contact%', 
            'Z-Contact%', 'Contact%', 'Zone%', 'F-Strike%', 'SwStr%'
        ]
        cols_to_invert = ['O-Swing%', 'SwStr%']
        
        # Step 1: Scale the data
        scaler = StandardScaler()
        self.data[plate_discipline_columns] = scaler.fit_transform(self.data[plate_discipline_columns])
        
        # Step 2: Invert 'lower is better' stats
        for col in cols_to_invert:
            self.data[col] = self.data[col] * -1  
        
        # Step 3: PCA with more components
        pca = PCA(n_components=3)  # Let's capture the top 3 PCs for demonstration
        X_pca = pca.fit_transform(self.data[plate_discipline_columns])
        
        # Extract PC1, PC2, PC3
        PC1 = X_pca[:, 0]
        PC2 = X_pca[:, 1]
        PC3 = X_pca[:, 2]
        
        # # Inspect explained variance
        # print("Explained variance ratio:", pca.explained_variance_ratio_)
        # print("Cumulative variance explained:", np.cumsum(pca.explained_variance_ratio_))
        
        # # Inspect loadings for interpretability
        # loadings = pca.components_
        # for i, component in enumerate(loadings):
        #     print(f"PC{i+1} loadings:")
        #     for col_name, weight in zip(plate_discipline_columns, component):
        #         print(f"  {col_name}: {weight:.4f}")
        #     print()

        # Step 4: Create a composite score
        # Example: 70% weight on PC1, 30% on PC2, ignoring PC3 or weighting it lightly.
        composite = 0.60 * PC1 + 0.40 * PC2 + 0.1 * PC3# You can adjust weights as you see fit

        # Step 5: Normalize composite to a 0–100 range
        composite_min, composite_max = composite.min(), composite.max()
        plate_disc_score = (composite - composite_min) / (composite_max - composite_min) * 100
        self.data["PlateDisciplineScore"] = plate_disc_score
    

    def filter_data(self):
        """
        Filters the data based on the player's position category.
        If the player's position category is BATTER, it selects specific columns related to batting statistics
        and updates the data attribute with these columns.
        Returns:
            None
        """

        if self.position_category == PositionCategory.BATTER:
            self.calc_plate_discipline_score()

            columns = ['PlayerName', 'Year', 'G', 'PA', 'AB', 'AVG', 'OBP', 'SLG', 'wOBA', 'wRC+', 'H', 'HBP',
                       '1B', '2B', '3B', 'HR', 'R', 'RBI', 'BB%', 'K%', 'ISO', 'SB', 'CS',
                       'xwOBA', 'xAVG', 'xSLG', 'EV', 'LA', 'Barrel%', 'HardHit%', 'PlateDisciplineScore', 'BaseRunning']
            
            self.data = self.data[columns]

        elif self.position_category == PositionCategory.SP:
            pass
        elif self.position_category == PositionCategory.RP:
            pass
        return
    
    def calc_fantasy_points(self):
        """
        Calculate fantasy points for each player in the dataset.
        This method calculates fantasy points for players based on their performance
        statistics and their position category. The points are calculated using the following formula
        based on Yahoo Fantasy Baseball scoring:
        - Single (1B): 2.6 points
        - Double (2B): 5.2 points
        - Triple (3B): 7.8 points
        - Home Run (HR): 10.4 points
        - Run (R): 1.9 points
        - Run Batted In (RBI): 1.9 points
        - Stolen Base (SB): 4.2 points
        - Hit By Pitch (HBP): 2.6 points
        The calculated points are stored in a new DataFrame which is then merged with
        the original dataset.
        Returns:
            pd.DataFrame: The original dataset with an additional column for total fantasy points.
        """

        fantasy_points = []

        for _, player in self.data.iterrows():
            if self.position_category == PositionCategory.BATTER:
                points = {
                    'PlayerName': player['PlayerName'],
                    'Year': player['Year'],  # Add Year to points dictionary
                    'TotalPoints': (
                        player['1B'] * 2.6 +
                        player['2B'] * 5.2 + 
                        player['3B'] * 7.8 +
                        player['HR'] * 10.4 +
                        player['R'] * 1.9 +
                        player['RBI'] * 1.9 +
                        player['SB'] * 4.2 +
                        player['HBP'] * 2.6
                    )
                }
                fantasy_points.append(points)
            elif self.position_category == PositionCategory.SP:
                pass
            elif self.position_category == PositionCategory.RP:
                pass
        points_df = pd.DataFrame(fantasy_points)
        self.data = pd.merge(self.data, points_df, on=['PlayerName', 'Year'], how='left')  # Merge on both PlayerName and Year
    
if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    batters = DataProcessing(PositionCategory.BATTER)
    batters.filter_data()
    
    with open("batter_data.txt", "w") as f:
        print(batters.data, file=f)


    batters.calc_fantasy_points()
    with open("batter_points.txt", "w") as f:
        print(batters.data, file=f)
