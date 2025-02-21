import pytest
import requests
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
def test_get_data(mock_get_page, scraper):
    mock_get_page.return_value = [{"player": "Player1", "stat": "Stat1"}]
    df = scraper.get_data()
    assert not df.empty
    assert df.iloc[0]['player'] == 'Player1'
    assert df.iloc[0]['stat'] == 'Stat1'


