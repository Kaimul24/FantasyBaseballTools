import pytest
import requests
import os
import pandas as pd
from FangraphsScraper.fangraphsScraper import FangraphsScraper, PositionCategory
from unittest.mock import patch

@pytest.fixture(params=[PositionCategory.BATTER, PositionCategory.SP, PositionCategory.RP])
def scraper(request):
    return FangraphsScraper(request.param)

def test_invalid_position_category():
    with pytest.raises(ValueError, match="Invalid Position Category"):
        FangraphsScraper("INVALID")
    
@patch('FangraphsScraper.fangraphsScraper.requests.get')
def test_get_page_no_respone(mock_get, scraper):
    mock_response = requests.Response()
    mock_response.status_code = 404
    mock_get.return_value = mock_response
    stats = scraper._get_page(2024)
    assert(not stats)


@patch('FangraphsScraper.fangraphsScraper.requests.get')
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

@patch('FangraphsScraper.fangraphsScraper.FangraphsScraper._get_page')
@patch('os.path.exists')
def test_get_data_no_player_data(mock_path_exists, mock_get_page, scraper):
    # Make sure we don't use cached files
    mock_path_exists.return_value = False
    
    mock_get_page.return_value = [{"player": "Player1", "stat": "Stat1"}]
    df = scraper.get_data()
    
    assert not df.empty
    # Check if the player column exists and has the expected value
    assert 'player' in df.columns
    assert df.iloc[0]['player'] == 'Player1'

@patch('FangraphsScraper.fangraphsScraper.FangraphsScraper._get_page')
@patch('os.path.exists')
@patch('pandas.read_pickle')
def test_get_data_with_player_data(mock_read_pickle, mock_path_exists, mock_get_page, scraper):
    # Set up the mock to simulate existing pickle file
    mock_path_exists.return_value = True
    
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
    
    # Verify read_pickle was called with the correct path format
    expected_path = os.path.join(
        "player_data", 
        f"{str(scraper.positionCategory)[17:]}_{scraper.start_year}_{scraper.end_year}.pkl"
    )
    mock_read_pickle.assert_called_once_with(expected_path)


