from lxml import html
import json
import requests
import pandas as pd
from enum import Enum

from config import DATA_DIR

class PositionCategory(Enum):
    BATTER = 1
    SP = 2
    RP = 3
 
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
    def __init__(self, position_category: PositionCategory, start_year: int = 2024, end_year: int = 2024) -> None:
        """
        Initialize a FangraphsScraper instance.
        
        Parameters
        ----------
        position_category : PositionCategory
            The category of players to scrape (BATTER, SP, or RP).
        start_year : int, optional
            The starting year for data collection, by default 2024.
        end_year : int, optional
            The ending year for data collection, by default 2024.
        
        Raises
        ------
        ValueError
            If an invalid position category is provided.
        """
        self.positionCategory = position_category
        self.start_year = start_year
        self.end_year = end_year

        self.base_urls = {
            PositionCategory.BATTER: 'https://www.fangraphs.com/leaders/major-league?pos=all&stats=bat&lg=all&type=8&season={}&month=0&season1={}&ind=0&pageitems=2000000000&qual=100',
            PositionCategory.SP: 'https://www.fangraphs.com/leaders/major-league?pos=all&lg=all&type=5&season={}&month=0&season1={}&ind=0&stats=sta&qual=60&pagenum=1&pageitems=2000000000',
            PositionCategory.RP: 'https://www.fangraphs.com/leaders/major-league?pos=all&lg=all&type=8&season={}&month=0&season1={}&ind=0&qual=20&pageitems=2000000000&stats=rel'
        } 
        
        if position_category not in self.base_urls:
            raise ValueError("Invalid Position Category")

    def _get_page(self, year: int) -> list:
        """
        Fetches and parses the HTML content of the page specified by the instance's URL.
        
        This method sends a GET request to the URL, parses the HTML content to extract
        a JSON script, and navigates through the JSON structure to retrieve player data.
        
        Parameters
        ----------
        year : int
            The year for which to fetch data.
            
        Returns
        -------
        list
            A list of statistical data extracted from the page. If no relevant
            data is found, an empty list is returned.
            
        Raises
        ------
        ValueError
            If JSON parsing fails or if no relevant query is found in the data.
        """
        self.url = self.base_urls[self.positionCategory].format(year, year)
        response = requests.get(self.url)
        if response.status_code != 200:
            return []
        
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

        relevant_query = queries[0] if queries else None
        
        if not relevant_query:
            raise ValueError("No relevant query found in the JSON data")
         
        state = relevant_query.get("state", {})
        data = state.get("data", {})
        stats = data.get("data", [])
        return stats
    
    def get_data(self) -> pd.DataFrame:
        """
        Retrieve statistical data from Fangraphs.
        
        Calls _get_page() to retrieve the statistical data from the Fangraphs page.
        If data for the requested years has been previously saved, it will be loaded
        from disk instead of re-scraping.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the statistical data.
        
        Raises
        ------
        ValueError
            If no data is found for any year in the specified range.
        """
        all_years_file_path = DATA_DIR / f"{str(self.positionCategory)[17:]}_{self.start_year}_{self.end_year}.pkl"
        if all_years_file_path.exists():
            return pd.read_pickle(all_years_file_path)
        
        all_data = []
        for year in range(self.start_year, self.end_year + 1):
            single_year_path = DATA_DIR / f"{str(self.positionCategory)[17:]}_{year}.pkl"
            if single_year_path.exists():
                df = pd.read_pickle(single_year_path)
                all_data.append(df)
                continue

            stats_data = self._get_page(year)
            if stats_data:
                df = pd.DataFrame(stats_data)
                df['Year'] = year
                df = df.dropna(axis=1, how='all')
                df = df.loc[:, df.notna().any()]
                all_data.append(df)
                df.to_pickle(single_year_path)
        
        if not all_data:
            raise ValueError("No data found for any year")
        
        common_columns = set.intersection(*[set(df.columns) for df in all_data])
        all_data = [df[list(common_columns)] for df in all_data]
        
        all_dfs = pd.concat(all_data, ignore_index=True)
        all_dfs.to_pickle(all_years_file_path)
        return all_dfs

if __name__ == "__main__":
    scraper = FangraphsScraper(PositionCategory.SP, 2019, 2019)    
    df = scraper.get_data()    
    cols = list(df.columns)
    with open("cols_SP.txt", "w") as f:        
        print(cols, file=f)
