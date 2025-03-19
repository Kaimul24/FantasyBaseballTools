import pytest
import requests
import os
import pandas as pd
from src.FangraphsScraper.fangraphsScraper import FangraphsScraper, PositionCategory
from unittest.mock import patch

@pytest.fixture(params=[PositionCategory.BATTER, PositionCategory.SP, PositionCategory.RP])
def scraper(request):
    return FangraphsScraper(request.param)

def test_invalid_position_category():
    with pytest.raises(ValueError, match="Invalid Position Category"):
        FangraphsScraper("INVALID")
    
@patch('src.FangraphsScraper.fangraphsScraper.requests.get')
def test_get_page_no_respone(mock_get, scraper):
    mock_response = requests.Response()
    mock_response.status_code = 404
    mock_get.return_value = mock_response
    stats = scraper._get_page(2024)
    assert(not stats)


@patch('src.FangraphsScraper.fangraphsScraper.requests.get')
def test_get_page(mock_get, scraper):
    mock_response = requests.Response()
    mock_response.status_code = 200
    mock_response._content = b'''
    <html>
        <script id="__NEXT_DATA__" type="application/json">
        {"props":{"pageProps":{"dehydratedState":{"queries":[{"state":{"data":{"data":[{"player":"Player1","stat":"Stat1"}]}}}]}}}}
        </script>
    </html>
    '''
    mock_get.return_value = mock_response

    stats = scraper._get_page(2024)
    assert len(stats) == 1
    assert stats[0]['player'] == 'Player1'
    assert stats[0]['stat'] == 'Stat1'

@patch('src.FangraphsScraper.fangraphsScraper.FangraphsScraper._get_page')
@patch('src.FangraphsScraper.fangraphsScraper.DATA_DIR')
def test_get_data_no_player_data(mock_data_dir, mock_get_page, scraper):
    # Create a mock Path object
    mock_path = mock_data_dir / f"{str(scraper.positionCategory)[17:]}_{scraper.start_year}_{scraper.end_year}.pkl"
    mock_year_path = mock_data_dir / f"{str(scraper.positionCategory)[17:]}_{scraper.start_year}.pkl"
    
    # Make exists() return False for our paths
    mock_path.exists.return_value = False
    mock_year_path.exists.return_value = False
    
    # Use the correct column names in your mock data
    # These should match what's in your real scraper's _get_page output
    mock_get_page.return_value = [{"PlayerNameRoute": "Player1", "HR": 10, "AVG": 0.300}]
    
    df = scraper.get_data()
    
    assert not df.empty
    # Check for the columns your scraper actually returns
    assert 'PlayerNameRoute' in df.columns
    assert df.iloc[0]['PlayerNameRoute'] == 'Player1'

@patch('src.FangraphsScraper.fangraphsScraper.FangraphsScraper._get_page')
@patch('src.FangraphsScraper.fangraphsScraper.DATA_DIR')
@patch('pandas.read_pickle')
def test_get_data_with_player_data(mock_read_pickle, mock_data_dir, mock_get_page, scraper):
    # Create a mock Path object that matches what pathlib creates
    mock_path = mock_data_dir / f"{str(scraper.positionCategory)[17:]}_{scraper.start_year}_{scraper.end_year}.pkl"
    
    # Make exists() return True for our mock path
    mock_path.exists.return_value = True
    
    # Create test dataframe that will be "loaded" from pickle
    test_df = pd.DataFrame({
        'player': ['Player1', 'Player2'],
        'stat': [10, 20],
        'Year': [2024, 2024]
    })
    mock_read_pickle.return_value = test_df
    
    # Call the method under test
    df = scraper.get_data()
    
    # Verify the dataframe matches our mock data
    assert not df.empty
    assert 'player' in df.columns
    assert df.iloc[0]['player'] == 'Player1'
    
    # Verify _get_page was NOT called since we loaded from pickle
    mock_get_page.assert_not_called()
    
    # Verify read_pickle was called with our mock path
    mock_read_pickle.assert_called_once_with(mock_path)


