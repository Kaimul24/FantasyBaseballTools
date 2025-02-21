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
    def __init__(self, position_category: PositionCategory, start_year: int = 2024, end_year: int = 2024):
        self.positionCategory = position_category
        self.start_year = start_year
        self.end_year = end_year

        

        ## QUAL TO BE CHANGED
        self.base_urls = {
            PositionCategory.BATTER: 'https://www.fangraphs.com/leaders/major-league?pos=all&stats=bat&lg=all&type=8&season={}&month=0&season1={}&ind=0&pageitems=2000000000&qual=100',
            PositionCategory.SP: 'https://www.fangraphs.com/leaders/major-league?pos=all&lg=all&qual=y&type=8&season={}&month=0&season1={}&ind=0&pageitems=2000000000&stats=sta',
            PositionCategory.RP: 'https://www.fangraphs.com/leaders/major-league?pos=all&lg=all&qual=y&type=8&season={}&month=0&season1={}&ind=0&pageitems=2000000000&stats=rel'
        }
        
        if position_category not in self.base_urls:
            raise ValueError("Invalid Position Category")

    def _get_page(self, year: int)-> list: 
        """
        Fetches and parses the HTML content of the page specified by the instance's URL.
        This method sends a GET request to the URL, parses the HTML content to extract
        a JSON script, and navigates through the JSON structure to retrieve player data.
        Returns:
            list: A list of statistical data extracted from the page. If no relevant
                  data is found, an empty list is returned.
        """
        self.url = self.base_urls[self.positionCategory].format(year, year)
        response = requests.get(self.url)
        if response.status_code != 200:
            return []  # Return empty list instead of raising exception
        
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
        all_data = []
        for year in range(self.start_year, self.end_year + 1):
            stats_data = self._get_page(year)
            if stats_data:
                df = pd.DataFrame(stats_data)
                df['Year'] = year
                # Remove columns with all NA values
                df = df.dropna(axis=1, how='all')
                # Remove empty columns
                df = df.loc[:, df.notna().any()]
                all_data.append(df)
        
        if not all_data:
            raise ValueError("No data found for any year")
        
        # Ensure all DataFrames have same columns
        common_columns = set.intersection(*[set(df.columns) for df in all_data])
        all_data = [df[list(common_columns)] for df in all_data]
        
        return pd.concat(all_data, ignore_index=True)

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    scraper = FangraphsScraper(PositionCategory.BATTER, 2024, 2024)    
    df = scraper.get_data()    
    with open("data.txt", "w") as f:        
        print(df, file=f)