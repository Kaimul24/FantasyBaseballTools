from FangraphsScraper.fangraphsScraper import FangraphsScraper, PositionCategory
import pandas as pd
import sklearn as sk

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



class DataProcessing:
    
    def __init__(self, position_category: PositionCategory):
        self.position_category = position_category
        self.data = FangraphsScraper(position_category).get_data()
    

    def filter_data(self):
        """
        Filters the data based on the player's position category.
        If the player's position category is BATTER, it selects specific columns related to batting statistics
        and updates the data attribute with these columns.
        Returns:
            None
        """

        if self.position_category == PositionCategory.BATTER:
            columns = ['PlayerName', 'G', 'PA', 'AB', 'AVG', 'OBP', 'SLG', 'wOBA', 'wRC+', 'H', 'HBP',
                       '1B', '2B', '3B', 'HR', 'R', 'RBI', 'BB%', 'K%', 'ISO', 'SB', 'CS',
                       'xwOBA', 'xAVG', 'xSLG', 'EV', 'LA', 'Barrel%', 'HardHit%']
            
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
        self.data = pd.merge(self.data, points_df, on='PlayerName', how='left')
        

    
    
    
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
