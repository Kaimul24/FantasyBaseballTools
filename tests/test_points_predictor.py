# import pytest
# import pandas as pd
# import numpy as np
# from unittest.mock import Mock, patch
# from PointsPredictor3 import load_data, create_rolling_datasets, concat_training_windows

# @pytest.fixture
# def sample_data():
#     data = {
#         'PlayerName': ['Player1', 'Player2', 'Player3'] * 4,
#         '2019_AVG': [.300, .275, .250] + [np.nan] * 9,
#         '2019_HR': [20, 15, 10] + [np.nan] * 9,
#         '2020_AVG': [np.nan] * 3 + [.290, .280, .270] + [np.nan] * 6,
#         '2020_HR': [np.nan] * 3 + [17, 20, 15] + [np.nan] * 6,
#         '2021_AVG': [np.nan] * 3 + [.280, .290, .310] + [np.nan] * 6,
#         '2021_HR': [np.nan] * 3 + [18, 22, 25] + [np.nan] * 6,
#         '2022_AVG': [np.nan] * 6 + [.285, .295, .270] + [np.nan] * 3,
#         '2022_HR': [np.nan] * 6 + [19, 21, 17] + [np.nan] * 3,
#         '2023_AVG': [np.nan] * 9 + [.290, .278, .305],
#         '2023_HR': [np.nan] * 9 + [22, 18, 24]
#     }
#     return pd.DataFrame(data)

# @pytest.fixture
# def mock_data_processing(sample_data):
#     """Mock DataProcessing class"""
#     mock = Mock()
#     mock.data = sample_data
#     return mock

# def test_load_data(mock_data_processing):
#     """Test load_data function"""
#     with patch('PointsPredictor3.DataProcessing', return_value=mock_data_processing):
#         df = load_data()
#         assert isinstance(df, pd.DataFrame)
#         assert 'PlayerName' in df.columns
#         assert not df.empty

# def test_create_rolling_datasets(sample_data):
#     """Test create_rolling_datasets function with dynamic year detection"""
#     datasets = create_rolling_datasets(sample_data)
    
#     # Get actual years from sample data (excluding 2020)
#     available_years = sorted(list(set(
#         int(col.split('_')[0]) 
#         for col in sample_data.columns 
#         if '_' in col and col.split('_')[0].isdigit() and col.split('_')[0] != '2020'
#     )))
    
#     # Calculate expected number of windows
#     expected_windows = len(available_years) - 3  # Need 4 consecutive years for each window
#     assert len(datasets) == expected_windows
    
#     # Test each window's structure
#     for i, dataset in enumerate(datasets):
#         # Years should be consecutive (excluding 2020)
#         assert dataset['train_years'] == available_years[i:i+2]  # First two years
#         assert dataset['val_year'] == available_years[i+2]      # Third year
#         assert dataset['test_year'] == available_years[i+3]     # Fourth year
        
#         # Check data splits
#         assert 'PlayerName' in dataset['train_data'].columns
#         assert 'PlayerName' in dataset['val_data'].columns
#         assert 'PlayerName' in dataset['test_data'].columns
        
#         # Verify correct columns are present in each split
#         train_years = dataset['train_years']
#         val_year = dataset['val_year']
#         test_year = dataset['test_year']
        
#         # Check training data contains only training years
#         train_cols = [col for col in dataset['train_data'].columns if '_' in col]
#         assert all(any(str(year) in col for year in train_years) for col in train_cols)
        
#         # Check validation data contains only validation year
#         val_cols = [col for col in dataset['val_data'].columns if '_' in col]
#         assert all(str(val_year) in col for col in val_cols)
        
#         # Check test data contains only test year
#         test_cols = [col for col in dataset['test_data'].columns if '_' in col]
#         assert all(str(test_year) in col for col in test_cols)

# def test_concat_training_windows(sample_data):
#     """Test concat_training_windows function"""
#     # First create the datasets using create_rolling_datasets
#     datasets = create_rolling_datasets(sample_data)
    
#     # Process the windows
#     processed_windows = concat_training_windows(datasets)
    
#     # Test the structure and calculations
#     for window in processed_windows:
#         # Check required keys exist
#         assert set(['train_years', 'val_year', 'test_year', 'weighted_data', 
#                    'train_data', 'val_data', 'test_data']).issubset(window.keys())
        
#         # Check weighted_data structure
#         weighted_df = window['weighted_data']
#         assert 'PlayerName' in weighted_df.columns
#         assert not weighted_df.empty
        
#         # Test weighting calculations for a specific stat (e.g., AVG)
#         year1, year2 = window['train_years']
        
#         if f'{year1}_AVG' in window['train_data'].columns and f'{year2}_AVG' in window['train_data'].columns:
#             # Get original values for a player that exists in both years
#             player_mask = ~window['train_data'][f'{year1}_AVG'].isna() & ~window['train_data'][f'{year2}_AVG'].isna()
#             if player_mask.any():
#                 player = window['train_data'].loc[player_mask, 'PlayerName'].iloc[0]
                
#                 year1_val = window['train_data'].loc[
#                     window['train_data']['PlayerName'] == player, f'{year1}_AVG'].iloc[0]
#                 year2_val = window['train_data'].loc[
#                     window['train_data']['PlayerName'] == player, f'{year2}_AVG'].iloc[0]
                
#                 # Calculate expected weighted value (0.4 * year1 + 0.6 * year2)
#                 expected_weighted = (0.4 * year1_val + 0.6 * year2_val)
                
#                 # Get actual weighted value
#                 actual_weighted = weighted_df.loc[
#                     weighted_df['PlayerName'] == player, 'weighted_AVG'].iloc[0]
                
#                 # Compare with small tolerance for floating point arithmetic
#                 assert abs(expected_weighted - actual_weighted) < 1e-10

# if __name__ == '__main__':
#     pytest.main([__file__])
