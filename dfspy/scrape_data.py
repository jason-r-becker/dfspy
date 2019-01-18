import argparse
import os
import sys
from datetime import datetime as dt
from datetime import timedelta
from time import sleep

import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

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
    
    # Scrape and save data.
    scraper = dataScraper(
        sources=args.sources,
        weeks=args.weeks,
        years=args.years,
        )
    scraper.save()

def parse_args():
    """Collect settings from command line and set defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sources', help='Data Sources')
    parser.add_argument('-w', '--weeks', help='Week(s) of the season to scrape')
    parser.add_argument('-y', '--years', help='Year(s) of seasons to scrape.')
    
    # Set default sources.
    # TODO: Update sources as their scrapers are written.
    default_sources = ['CBS', 'ESPN', 'Yahoo']
    
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
        weeks=default_week,
        years=default_year,
        )
    args = parser.parse_args()
    
    # Convert ranges to lists.
    years, weeks = args.years, args.weeks
    args.years = list_from_str_range(years) if '-' in years else years
    args.weeks = list_from_str_range(weeks) if '-' in weeks else weeks
    
    return args

def list_from_str_range(str_range):
    """Convert str range to inclusive list of integers"""
    vals = str_range.split('-')
    return list(range(int(vals[0]), int(vals[1])+1))
    
    
def mkdir(directory):
    """Make directory if it does not already exist."""
    try:
        os.makedirs(directory)
    except OSError:
        pass
        
        
class dataScraper:
    """
    Class for scraping weekly or season projections from various
    web sources and saving to the `../data` directory.
    
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
    
    Methods
    -------
    save(): Scrape and save selected data.
    """
    
    def __init__(self, sources, weeks, years):
        # Convert inits to lists
        self.sources = sources if type(sources) in [list, tuple] else [sources]
        self.years = years if type(years) in [list, tuple] else [years]
        self.positions = 'QB RB WR TE DST K'.split()
        if weeks == 'all':
            self.weeks = np.arange(17)
        elif weeks == 'season':
            self.weeks = [0]
        elif type(weeks) in [list, tuple]:
            self.weeks = weeks
        else:
            self.weeks = [weeks]
            
    def save(self):
        """Scrape and save specified data."""
        iters = len(self.years) * len(self.weeks) * len(self.positions) \
                * len(self.sources)
        with tqdm(total=iters) as pbar:
            for year in self.years:
                for week in self.weeks:
                    for pos in self.positions:
                        for source in self.sources:
                            scraper = singleScrape(source, week, year, pos)
                            df = scraper.scrape()
                            if df is not None:
                                path = f'../data/{year}/{week}/{pos}'
                                mkdir(path)
                                df.to_csv(f'{path}/{source}.csv', index=False)
                            pbar.update(1)


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
        
    def _clean_CBS_df(self, df):
        header_ix = list(df.iloc[:, 0]).index('Player')
        cols = df.iloc[header_ix, :]
        clean_df = df.iloc[header_ix+1:-1, :].copy()
        clean_df.columns = cols
        clean_df.index = list(range(len(clean_df)))
        clean_df['POS'] = self.pos
        return clean_df
        
    def _url_for_ESPN(self):
        pos = {'QB': 0, 'RB': 2, 'WR': 4, 'TE': 6, 'DST': 16, 'K': 17}
        cat_id = f'&slotCategoryId={pos[self.pos]}'
        period = 'seasonTotals=true' if self.week == 0 \
                 else f'scoringPeriodId={self.week}'
        page = f'&startIndex={int(self._page * 40)}'
        url = (
            f'http://games.espn.com/ffl/tools/projections?projections?='
            f'slot=CategoryId=0{cat_id}&{period}&seasonId={self.year}{page}'
            )
        return url
        
    def _clean_ESPN_df(self, df):
        header_ix = list(df.iloc[:, 0]).index('PLAYER, TEAM POS')
        cols = [col for col in df.iloc[header_ix, :] if isinstance(col, str)]
        clean_df = df.iloc[header_ix+1:, :len(cols)].copy()
        cols[0] = 'Player'
        clean_df.columns = cols
        
        clean_df.index = list(range(len(clean_df)))
        clean_df['POS'] = self.pos
        return clean_df
    
    def _url_for_Yahoo(self):
        pos = 'DEF' if self.pos == 'DST' else self.pos
        period = f'S_PS_{self.year}' if self.week == 0 else f'S_PW_{self.week}'
        page = f'&count={int(self._page * 25)}'
        url = (
            f'https://football.fantasysports.yahoo.com/f1/48938/players?&sort='
            f'PTS&sdir=1&status=A&pos={pos}&stat1={period}&jsenabled=1{page}'
            )
        return url
        
    def _clean_Yahoo_df(self, df):
        df.columns = df.columns.droplevel(0)  # drop first multi-index row
        bad_cols = ['Unnamed', 'Fan Pts', 'Owner', 'GP*', 'Owned']
        cols = [col for col in df.columns \
                if not any(bc in col for bc in bad_cols)]
        clean_df = df[cols].copy()
        temp_cols = list(clean_df)
        temp_cols[0] = 'Player'
        clean_df.columns = temp_cols
        clean_df['POS'] = self.pos
        return clean_df
        
    def _scrape_single_page(self):
        """
        Scrape DataFrame for specified inits.
        
        Returns
        -------
        df: pd.DataFrame
            DataFrame of sinlge page scrape, None if scrape fails.
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
        table_ix = 1 if self.source in ['Yahoo'] else 0
        table = soup.find_all('table')[table_ix]
        df = pd.read_html(str(table))[0]
        return self._clean_scraped_df(df)
    
    def scrape(self):
        """
        Scrape DataFrame for specified inits. If source hosts projections
        tables on multiple pages, scrape all pages and combine into one
        DataFrame, otherwise scrape full table if hosted on one page.
        
        Returns
        -------
        df: pd.DataFrame
            DataFrame of full resulting scrape, None if scrape fails.
        """
        multi_scrape_sources = ['ESPN', 'Yahoo']
        if self.source in multi_scrape_sources:
            max_pages = 10
            df_pages = []
            for i in range(max_pages):
                self._page = i
                self.url = self._get_url()
                try:
                    df_i = self._scrape_single_page()
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as e:
                    break
                
                if not isinstance(df_i, pd.DataFrame):
                    break
                if len(df_i) < 4:
                    break

                # Store succesfully scraped DataFrame.
                df_pages.append(df_i)
                if i == 0:
                    # Save column headers from first page.
                    cols = df_i.columns
            
            if len(df_pages) == 0:
                # No succesfull scrapes.
                return None
            
            # Combine all succesfully scraped pages.
            combined_df = np.vstack(df_pages)
            df = pd.DataFrame(combined_df, columns=cols)
            df.drop_duplicates(inplace=True)
        
        else:
            # Only single scrape required for all data.
            self.url = self._get_url()
            df = self._scrape_single_page()
        
        return df
        
        


# %%

if __name__ == '__main__':
    main()
