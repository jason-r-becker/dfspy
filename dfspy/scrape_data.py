import sys
from datetime import datetime as dt
from datetime import timedelta
from time import sleep

import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

# %%

def main():
    """
    Scrape data for specified period. Data saved to
    `../data/<year>/<week>/<source>.csv`.
    
    TODO: implement dataScraping class usage
    """
    # Disable warnings for unverified requests.
    requests.packages.urllib3.disable_warnings()
    
    # Get command line arguments.
    args = parse_args()


def parse_args():
    """Collect settings from command line and set defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sources', help='Data Sources')
    parser.add_argument('-w', '--weeks', help='Week(s) of the season to scrape')
    parser.add_argument('-y', '--years', help='Year(s) of seasons to scrape.')
    
    # Set default sources.
    # TODO: Update sources as their scrapers are written.
    default_sources = ['CBS', 'ESPN']
    
    # Get default season year and week from current time.
    today = dt.utcnow()
    
    # Default year is previous season until August.
    default_year = today.year if today.month > 7 else today.year - 1
    season_start = {  # Date of Thursday night game starting each season.
        2015: '9/10/2015',
        2016: '9/8/2016',
        2017: '9/7/2017',
        2018: '9/6/2018',
        2019: '9/5/2019',
        }[default_year]
        
    # Default week is the current NFL season week, starting on Tuesdays.
    # Before the season default is week 1, and after the season it is 16.
    default_week = int(np.ceil(
        (today-pd.to_datetime(season_start)+timedelta(days=2)).total_seconds()
        / (3600 * 24 * 7)
        ))
    default_week = max(1, min(16, default_week))
    
    # Set default arguments.
    parser.set_defaults(
        sources=default_sources,
        week=default_week,
        year=default_year,
        )
    args = parser.parse_args()
    return args
    
    
def mkdir(directory):
    """Make directory if it does not already exist."""
    try:
        os.makedirs(directory)
    except OSError:
        pass
        
        
class dataScraper:
    """
    Class for scraping weekly or season projections from various
    web sources.
    
    TODO: call singleScrape and build data directory for given inputs.
    
    Parameters
    ----------
    sources: {str, list(str)}
        List of sources to scrape from.
            - str: Scrape projections from single source.
            - list(str): Scrape projections from multiple sources.
    weeks: {int, list(int), 'season', 'all'}
        List of weeks to scrape projections.
            - int: Scrape single week's projections (0 for full season).
            - lits(int): Scrape multiple week's projections.
            - 'season': Scrape full season projections.
            - 'all': Scrape all individual weeks and full season projections.
    years: {int, list(int)}
        List of years to scrape projections.
            - int: Scrape single year's projections.
            - lits(int): Scrape multiple year's projections.
    """
    
    def __init__(self, sources, weeks, years):
        self.sources = sources if type(sources) in [list, tuple] else [sources]
        self.weeks = weeks if type(weeks) in [list, tuple] else [weeks]
        self.years = years if type(years) in [list, tuple] else [years]
        self.positions = 'QB RB WR TE DST K'.split()
        

class singleScrape:
    """
    Class for perfoming single scrape
    
    TODO: add scraping for:
        - FantasyData
        - Fantasypros
        - FantasySharks
        - FFToday
        - FleaFlicker
        - NumberFire
        - Yahoo
        - FantasyFootballNerd
        - RTSports
        - Walterfootball
        - Any other sources
        
    Parameters
    ----------
    source: str
        Data source.
    week: int
        Week of the season to scrape.
    year: int
        Year of the season to scrape
    pos: str
        Position to scrape.
    
    Methods
    -------
    scrape(): Returns scraped data as DataFrame.
    """
    
    def __init__(self, source, week, year, pos):
        self.source = source
        self.week = week
        self.year = year
        self.pos = pos
        self.url = self._get_url()
    
    def _get_url(self):
        """Returns url for scraping."""
        return eval(f'self._url_for_{self.source}()')

    def _clean_scraped_df(self, df):
        """Returns cleaned DataFrame from raw scrape."""
        clean_df = eval(f'self._clean_{self.source}_df(df)')
        return clean_df
        
    def _url_for_CBS(self):
        week = '' if self.week == 0 else self.week
        url = (
            f'https://www.cbssports.com/fantasy/football/stats/sortable'
            f'/points/{self.pos}/standard/projections/{self.year}/{week}'
            f'?print_rows=9999'
            )
        return url
    
    def _url_for_ESPN(self):
        pos = {'QB': 0, 'RB': 2, 'WR': 4, 'TE': 6, 'DST': 16, 'K': 17}
        cat_id = f'&slotCategoryId={pos[self.pos]}'
        period = 'seasonTotals=true' if self.week == 0 \
                 else f'scoringPeriodId={self.week}'
        url = (
            f'http://games.espn.com/ffl/tools/projections?projections?='
            f'slot=CategoryId=0{cat_id}&{period}&seasonId={self.year}'
            )
        return url
        
    def _clean_CBS_df(self, df):
        header_ix = list(df.iloc[:, 0]).index('Player')
        cols = df.iloc[header_ix, :]
        clean_df = df.iloc[header_ix+1:-1, :].copy()
        clean_df.columns = cols
        clean_df.index = list(range(len(clean_df)))
        clean_df['POS'] = self.pos
        return clean_df
        
    def _clean_ESPN_df(self, df):
        header_ix = list(df.iloc[:, 0]).index('PLAYER, TEAM POS')
        cols = [col for col in df.iloc[header_ix, :] if isinstance(col, str)]
        clean_df = df.iloc[header_ix+1:, :len(cols)].copy()
        cols[0] = 'Player'
        clean_df.columns = cols
        clean_df.index = list(range(len(clean_df)))
        clean_df['POS'] = self.pos
        return clean_df
    
    def scrape(self):
        """
        Scrape DataFrame for specified inits.
        
        Returns
        -------
        df: pd.DataFrame
            DataFrame of resulting scrape, None if scrape fails.
        """
        # Scrape data, attempt 5 times before passing.
        attempts = 5
        for attempt in range(attempts):
            try:
                r = requests.get(self.url, verify=False, timeout=5)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                sleep(1)
                r = None
                continue
            else:
                break
        
        # Print Warning if scrape failed all attempts.
        if r is None:
            msg = (
                f'Warning: Scrape failed for {self.pos} from {self.source} '
                f'Week {self.week} - {self.year}.'
                )
            print(msg)
            return None
        
        # Convert scrape into DataFrame and clean.
        soup = BeautifulSoup(r.text, features='lxml')
        table = soup.find_all('table')[0]
        df = pd.read_html(str(table))[0]
        return self._clean_scraped_df(df)


if __name__ == "__main__":
    main()
            
