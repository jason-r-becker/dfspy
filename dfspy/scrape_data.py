import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime as dt
from datetime import timedelta
from itertools import chain
from time import sleep

import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# %%

def main():
    """
    Scrape data for specified period.
    
    Data saved to `../data/<year>/<week>/<position>/<source>.csv`.
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
    default_sources = ['CBS', 'ESPN', 'FantasyPros', 'FFToday',
                       'NFL', 'RTSports', 'Yahoo']
    
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
    default_week = max(1, min(17, default_week))
    
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
        
def scrape_NFL_json_col_names():
    """Get column names for NFL.com json files."""
    url = 'https://api.fantasy.nfl.com/v1/game/stats?format=json'
    # Scrape data, attempt 5 times before passing.
    attempts = 5
    for attempt in range(attempts):
        try:
            r = requests.get(url, verify=False, timeout=5)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            sleep(1)
            r = None
            continue
        else:
            break
        
    if (r is not None and
        r.status_code == 200 and
        'stats' in r.json()):
        # Succesfull scrape; get column names.
        col_names = {sub_dict['id']: sub_dict['shortName'] \
                     for sub_dict in r.json()['stats']}
        
        # Output to json file.
        with open('../data/.NFL_json_columns.json', 'w') as fid:
            json.dump(col_names, fid, indent=4, sort_keys=True)
        
class dataScraper:
    """
    Class for scraping weekly or season projections from various
    web sources and saving to the `../data` directory.
    
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
            self.weeks = np.arange(18)
        elif weeks == 'season':
            self.weeks = [0]
        elif type(weeks) in [list, tuple]:
            self.weeks = weeks
        else:
            self.weeks = [weeks]
        
        self._headers = {}
        with open('../data/.NFL_json_columns.json', 'r') as fid:
            self._headers['NFL'] = json.load(fid)
            self._headers['STATS'] = self._headers['NFL']
            
    def _validate_current_period(self, source, week, year, pos):
        """Return True if data can be scraped else False."""
        week, year = int(week), int(year)
        if year < 2016:
            return False
        
        # No weekly histoirical Data.
        no_weekly_data = {
            'ESPN': [2016, 2017],
            'FantasyPros': [2016, 2017],
            'NFL': [2016, 2017],
            'RTSports': [2016, 2017, 2018],
            'Yahoo': [2016, 2017],
            }
        try:
            if week > 0 and year in no_weekly_data[source]:
                return False
        except KeyError:
            pass
        

        # No full season data.
        no_full_season_data = {
            'ESPN': [2016],
            'FantasyPros': [2016, 2017],
            'NFL': [2016, 2017],
            'RTSports': [2016, 2017],
            'Yahoo': [2016, 2017],
            }
        try:
            if week == 0 and year in no_full_season_data[source]:
                return False
        except KeyError:
            pass
        
        # No weekly DST for FFToday.
        if source == 'FFToday' and week > 0 and pos == 'DST':
            return False
        
        return True
        
    def save(self):
        """Scrape and save specified data."""
        iters = len(self.years) * len(self.weeks) * len(self.positions) \
                * len(self.sources)
        with tqdm(total=iters) as pbar:
            for year in self.years:
                for week in self.weeks:
                    for pos in self.positions:
                        for source in self.sources:
                            try:
                                header = self._headers[source]
                            except KeyError:
                                header = None
                            if not self._validate_current_period(
                                source, week, year, pos):
                                # Data does not exist.
                                pbar.update(1)
                                continue
                                
                            # Data exists; scrape and save.
                            scraper = singleScrape(
                                source, week, year, pos, header)
                            df = scraper.scrape()
                            if df is None:
                                msg = (
                                    f'Warning: Scrape failed for {pos} from '
                                    f'{source} Week {week} - {year}.'
                                    )
                                print(msg)
                            else:
                                path = f'../data/{year}/{week}/{pos}'
                                mkdir(path)
                                df.to_csv(f'{path}/{source}.csv', index=False)
                            pbar.update(1)

# %%

class singleScrape:
    """
    Class for perfoming single scrape
    
    TODO: add scraping for:
        - FantasyData
        - FantasySharks
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
    header: dict, default=None
        Dictionary conversion of headers for given source.
    Methods
    -------
    scrape(): Returns scraped data as DataFrame.
    """
    
    def __init__(self, source, week, year, pos, header=None):
        self.source = source
        self.week = week
        self.year = year
        self.pos = pos
        self.header = header
        
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
            f'slot=CategoryId=0{cat_id}{page}&{period}&seasonId={self.year}'
            )
        return url
        
    def _clean_ESPN_df(self, df):
        try:
            header_ix = list(df.iloc[:, 0]).index('PLAYER, TEAM POS')
        except ValueError:
            header_ix = list(df.iloc[:, 1]).index('PLAYER, TEAM POS')

        cols = [col for col in df.iloc[header_ix, :] if isinstance(col, str)]
        clean_df = df.iloc[header_ix+1:, :len(cols)].copy()
        if 'RNK' in cols:
            cols[1] = 'Player'
            clean_df.columns = cols
            clean_df.drop('RNK', inplace=True, axis=1)
        else:
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
        
    def _url_for_FantasyPros(self):
        pos = self.pos.lower()
        period = 'draft' if self.week == 0 else self.week
        url = (
            f'https://www.fantasypros.com/nfl/projections/{pos}'
            f'.php?week={period}'
            )
        return url
        
    def _clean_FantasyPros_df(self, df):
        # Drop first multi-index row if it there are multiple levels.
        try:
            levels = len(df.columns.levels)
        except AttributeError:
            pass
        else:
            df.columns = df.columns.droplevel(0)
        df['POS'] = self.pos
        return df
        
    def _url_for_FFToday(self):
        pos = {'QB': 10, 'RB': 20, 'WR': 30, 'TE': 40, 'DST': 99, 'K': 80}
        period = '' if self.week == 0 else f'&GameWeek={self.week}&'
        wk = '' if self.week == 0 else ''
        url = (
            f'http://www.fftoday.com/rankings/player{wk}proj.php?'
            f'Season={self.year}{period}&PosID={pos[self.pos]}&LeagueID=1'
            f'&order_by=FFPts&sort_order=DESC&cur_page={self._page}'
        )
        return url
        
    def _clean_FFToday_df(self, df):
        try:
            header_ix = list(df.iloc[:, 1]).index('Player Sort First: Last:')
        except ValueError:
            header_ix = list(df.iloc[:, 1]).index('Team')
        cols = [col for col in df.iloc[header_ix, :] if isinstance(col, str)]

        clean_df = df.iloc[header_ix+1:, :len(cols)].copy()
        cols[1] = 'Player'
        clean_df.columns = cols
        clean_df.drop('Chg', inplace=True, axis=1)
        return clean_df
    
    def _url_for_RTSports(self):
        pos = {'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3, 'DST': 5, 'K': 4}[self.pos]
        url = (
            f'https://www.freedraftguide.com/football/draft-guide-rankings'
            f'-provider.php?POS={pos}'
            )
        return url

    def _get_RTSports_df(self, raw_dict):
        """Get RTSports from json web source and convert to DataFrame."""
        # Get json keys for top and 'stats' levels of json.
        bad_keys = ['name', 'year', 'team', 'games_played']
        level_1_keys = ['name', 'nfl_team', 'college', 'experience',
                         'draft_year', 'draft_round', 'draft_pick']
        level_2_keys = [key for key in raw_dict['1']['stats'].keys() \
                         if key not in bad_keys]
        
        # Build dictionary of specified stats.
        clean_dict = defaultdict(list)
        for key in raw_dict.keys():
            for sub_key in level_1_keys:
                clean_dict[sub_key].append(raw_dict[key][sub_key])
            for sub_key in level_2_keys:
                clean_dict[sub_key].append(raw_dict[key]['stats'][sub_key])

        # Convert to DataFrame and clean.
        df = pd.DataFrame(clean_dict)
        df.rename(columns={'name': 'Player'}, inplace=True)
        df['POS'] = self.pos
        return df
        
    def _url_for_NFL(self):
        pos = 'DEF' if self.pos == 'DST' else self.pos
        period = f'seasonProjectedStats&season={self.year}' if self.week == 0 \
                 else f'weekProjectedStats&season={self.year}&week={self.week}'
        url = (
            f'http://api.fantasy.nfl.com/v1/players/stats?statType={period}'
            f'&position={pos}&format=json'
            )
        return url

    def _url_for_STATS(self):
        pos = 'DEF' if self.pos == 'DST' else self.pos
        period = f'seasonStats&season={self.year}' if self.week == 0 \
                 else f'weekStats&season={self.year}&week={self.week}'
        url = (
            f'http://api.fantasy.nfl.com/v1/players/stats?statType={period}'
            f'&position={pos}&format=json'
            )
        return url
        
    def _get_NFL_df(self, raw_dict):
        """Scrape NFL.com projections from json and convert to DataFrame."""
        # Find all unique stats which occur in given json.
        unique_cols_ix = list(set(chain(
            *[list(sub_dict['stats'].keys()) for sub_dict in raw_dict])))
        cols_ix = list(map(str, sorted(map(int, unique_cols_ix))))
        cols = [self.header[ix] for ix in cols_ix]
        
        # Find duplicate columns and remove them from list of stored columns.
        duplicate_cols = list(set([col for col in cols if cols.count(col) > 1]))
        for dup_col in duplicate_cols:
            ix = cols.index(dup_col)
            cols.pop(ix)
            cols_ix.pop(ix)
        
        # Build dictionary of specified stats.
        clean_dict = defaultdict(list)
        for sub_dict in raw_dict:
            for sub_key in ['name', 'teamAbbr']:
                clean_dict[sub_key].append(sub_dict[sub_key])
            for col, ix in zip(cols, cols_ix):
                try:
                    clean_dict[col].append(float(sub_dict['stats'][ix]))
                except KeyError:
                    clean_dict[col].append(0)

        # Convert to DataFrame and clean.
        df = pd.DataFrame(clean_dict)
        df.rename(columns={'name': 'Player', 'teamAbbr': 'Team'}, inplace=True)
        df['POS'] = self.pos
        return df
        
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
        
        if r is None:
            return None
        
        if self.source == 'RTSports':
            try:
                df = self._get_RTSports_df(r.json()['player_list'])
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                df = None
            return df
        elif self.source in ['NFL', 'STATS']:
            try:
                df = self._get_NFL_df(r.json()['players'])
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                df = None
                print(sys.exc_info()[0])
                print(e)
            return df
        
        # Convert scrape into DataFrame and clean.
        soup = BeautifulSoup(r.text, features='lxml')
        if self.source in ['Yahoo']:
            table_ix = 1
        elif self.source in ['FFToday']:
            table_ix = 3
        else:
            table_ix = 0
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
        multi_scrape_sources = ['ESPN', 'Yahoo', 'FFToday']
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
                    # print(e)
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




        

    
