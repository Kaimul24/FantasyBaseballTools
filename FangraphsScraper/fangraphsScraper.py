from lxml import html
import json
import requests
import pandas as pd
from enum import Enum

class PositionCategory(Enum):
    BATTER = 'Batter'
    SP = 'SP'
    RP = 'RP'
 
class FangraphsScraper:
    '''
    A scraper class for extracting statistics from Fangraphs.

    Methods
    -------
    _get_page():
        Fetches the webpage content, parses the JSON data embedded in the page,
        and extracts the relevant statistics.

    get_data():
        Retrieves the statistics data and returns it as a pandas DataFrame.

    '''

    def __init__(self, position_category: PositionCategory): # In the future, add parameters to customize the URL such as season, qualified PA, pitchers, etc.
        self.positionCategory = position_category
        if position_category == PositionCategory.BATTER:
            self.url = 'https://www.fangraphs.com/leaders/major-league?pos=all&stats=bat&lg=all&type=8&season=2024&month=0&season1=2024&ind=0&pageitems=2000000000&qual=y'
        
        elif position_category == PositionCategory.SP:
            self.url = 'https://www.fangraphs.com/leaders/major-league?pos=all&lg=all&qual=y&type=8&season=2024&month=0&season1=2024&ind=0&pageitems=2000000000&stats=sta'

        elif position_category == PositionCategory.RP:
            self.url = 'https://www.fangraphs.com/leaders/major-league?pos=all&lg=all&qual=y&type=8&season=2024&month=0&season1=2024&ind=0&pageitems=2000000000&stats=rel'
        else:
            raise ValueError("Invalid Position Category")

    def _get_page(self)-> list: 
        """
        Fetches and parses the HTML content of the page specified by the instance's URL.
        This method sends a GET request to the URL, parses the HTML content to extract
        a JSON script, and navigates through the JSON structure to retrieve player data.
        Returns:
            list: A list of statistical data extracted from the page. If no relevant
                  data is found, an empty list is returned.
        """
        response = requests.get(self.url)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch data: HTTP {response.status_code}")
        
        tree = html.fromstring(response.text)

        try:
            script_content = tree.xpath('//script[@id="__NEXT_DATA__"]/text()')
            json_data = json.loads(script_content[0])
        except (IndexError, json.JSONDecodeError) as e:
            raise ValueError("Failed to parse JSON data from the response") from e

        props = json_data.get("props", {})
        pageProps = props.get("pageProps", {})
        dehydratedState = pageProps.get("dehydratedState", {})
        queries = dehydratedState.get("queries", [])

        if queries:
            relevant_query = queries[0]
        
        if not relevant_query:
            raise ValueError("No relevant query found in the JSON data")
         
        state = relevant_query.get("state", {})
        data = state.get("data", {})
        stats = data.get("data", [])
        return stats
    
    """
    Calls _get_page() to retrieve the statistical data from the Fangraphs page.
    Returns:
        pandas.DataFrame: A DataFrame containing the statistical data.
    """
    def get_data(self) -> pd.DataFrame:
        stats_data = self._get_page()
        if not stats_data:
            raise ValueError("No data found, check the URL.")
        df = pd.DataFrame(stats_data)
        return df

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    scraper = FangraphsScraper(PositionCategory.BATTER)
    df = scraper.get_data()
    with open("data.txt", "w") as f:
        print(df, file=f)
