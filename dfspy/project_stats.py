import warnings
from collections import defaultdict
from glob import glob

import fancyimpute as fi
import numpy as np
import pandas as pd


# %%
def impute(df, method, verbose=False):
    """
    Impute missing data using specified imputation method.
    
    Parameters
    ----------
    df: pd.DataFrame
        Stat DataFrame with source columns and player/team  multi-index.
    method: str/bool
        Imputation method for missing data.
            - False: Do not impute missing data.
            - None: Do not impute missing data.
            - 'BiScaler'
            - 'IterativeImpute'
            - 'IterativeSVD'
            - 'KNN': Impute with nearest neighbors.
            - 'Mean': Impute missing with average of other sources.
            - 'NuclearNorm'
            - 'SoftImpute'
    verbose: bool, default=False
        If True, print debugging information.
        
    Returns
    -------
    df: pd.DataFrame
        Imputed DataFrame with no NaNs.
    """
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # Subset DataFrame to only include only projection columns.
    ignored_cols = ['Player', 'Team', 'Pos', 'Week', 'STATS']
    impute_cols = [col for col in list(df) if col not in ignored_cols]
    X = df[impute_cols].copy().T
    
    # Impute DataFrame.
    v = verbose
    if method in [None, False]:
        imputed_vals = X.values
    elif np.sum(np.sum(X.isnull())) == 0:
        # No missing values.
        imputed_vals = X.values
    elif method == 'BiScaler':
        imputed_vals = fi.BiScaler(verbose=v).fit_transform(X)
    elif method == 'IterativeImpute':
        imputed_vals = fi.IterativeImputer(verbose=v).fit_transform(X)
    elif method == 'IterativeSVD':
        imputed_vals = fi.IterativeSVD(verbose=v).fit_transform(X)
    elif method == 'KNN':
        imputed_vals = fi.KNN(k=3, verbose=v).fit_transform(X)
    elif method == 'MatrixFactorization':
        imputed_vals = fi.MatrixFactorization(verbose=v).fit_transform(X)
    elif method == 'Mean':
        imputed_vals = fi.SimpleFill('mean').fit_transform(X)
    elif method == 'Median':
        imputed_vals = fi.SimpleFill('median').fit_transform(X)
    elif method == 'NuclearNorm':
        imputed_vals = fi.NuclearNormMinimization(verbose=v).fit_transform(X)
    elif method == 'SoftImpute':
        imputed_vals = fi.SoftImpute(verbose=v).fit_transform(X)
    
    # Recombine ignored columns with imputed data.
    imputed_df = pd.DataFrame(imputed_vals.T, columns=X.index)
    for col in impute_cols:
        if len(imputed_df[col]) != len(df[col]):
            print(f'df: {len(df[col])}\nimp: {len(imputed_df[col])}')
        df[col] = imputed_df[col].values
    
    return df


def impute_realized_stats(df, method):
    """
    Impute realized STATS column of DataFrame if it exists.
    
    Parameters
    ----------
    df: pd.DataFrame
        Stat DataFrame with source columns and player/team  multi-index.
    method: {0, 'drop', False, None}
        Imputing method for NaN's in STATS column.
            - 0: Replace with 0.
            - drop: Drop rows wiht NaN values.
            - False: Return df with NaN values.
            - None: Return df with NaN values.
    Returns
    -------
    df: pd.DataFrame
        Imputed DataFrame with no NaNs in STATS column.
    """
    if 'STATS' in list(df):
        if method is False:
            pass
        if method is None:
            pass
        elif method == 0:
            df['STATS'].fillna(0, inplace=True)
        elif method == 'drop':
            drop_ix = df['STATS'].isnull()
            df = df[~drop_ix].copy()
    
    return df
    
    
class TrainProjections:
    """
    
    Parameters
    ----------
    pos: str
        Player position.
    year: int
        Year of the season.
    season: bool, default=False
        If True use season projections, else combine weekly projections.
        
    Attribues
    ---------
    stats: list[str]
        Statistics to be projected for specified postion.
    
    Methods
    -------
    load_data(weeks): Load and impute projections for specified weeks.
        Stat DataFrames are saved in self.stat_dfs dict with stat name keys.
    read_data(stat_dfs): Read data from dict.
    """
        
    def __init__(self, pos, year=2018, season=False):
        self.pos = pos
        self.year = year
        self.season = season
    
        # Store relevant stats.
        if self.pos == 'DST':
            self.stats = 'PA.YdA.TD.Sack.Saf.Int.Fum Rec.Blk'.split('.')
        else:
            self.stats =  [
                'Pass Yds', 'Pass TD', 'Pass Int', 'Rush Yds', 'Rush TD',
                'Receptions', 'Rec Yds', 'Rec TD', '2PT',
                ]
        self.essential_stats = {
            'QB': ['Pass Yds', 'Pass TD', 'Pass Int', 'Rush Yds', 'Rush TD'],
            'RB': ['Rush Yds', 'Rush TD', 'Receptions', 'Rec Yds', 'Rec TD'],
            'WR': ['Rush Yds', 'Rush TD', 'Receptions', 'Rec Yds', 'Rec TD'],
            'TE': ['Receptions', 'Rec Yds', 'Rec TD'],
            'DST': ['PA', 'YdA', 'TD', 'Sack', 'Int', 'Fum Rec'],
            }
        self.nonessential_stats = {
            'QB': ['Receptions', 'Rec Yds', 'Rec TD', '2PT'],
            'RB': ['Pass Yds', 'Pass TD', 'Pass Int', '2PT'],
            'WR': ['Pass Yds', 'Pass TD', 'Pass Int', '2PT'],
            'TE': ['Pass Yds', 'Pass TD', 'Pass Int',
                   'Rush Yds', 'Rush TD', '2PT'],
            'DST': ['Saf', 'Blk'],
            }
        
    def load_data(self, weeks, proj_impute_method='IterativeImpute',
                  stat_impute_method=0):
        """
        Load data for given weeks and impute missing data.
        
        Parameters
        ----------
        weeks: int/range/list[int]
            Weeks of the season to include in DataFrame.
        proj_impute_method: str/bool, default='IterativeImpute'
            Imputation method for missing data.
                - False: Do not impute missing data.
                - 'BiScaler'
                - 'IterativeImpute'
                - 'IterativeSVD'
                - 'KNN': Impute with nearest neighbors.
                - 'Mean': Impute missing with average of other sources.
                - 'NuclearNorm'
        stat_impute_method: {0, 'drop'}, default=0
            Imputation for missing data in realized STATS column.
            
        """
        self.proj_impute_method = proj_impute_method
        self.stat_impute_method = stat_impute_method
        
        # Convert weeks to appropriate range.
        if isinstance(weeks, int):
            weeks = [weeks]
        elif weeks == 'training':
            weeks = range(1, 11)
        elif weeks == 'validation':
            weeks = range(11, 15)
        elif weeks == 'test':
            weeks = range(15, 18)
        else:
            weeks = list(weeks)
        
        # Combine stats from specified weeks into a dictionary with stat names
        # as keys for DataFrames with projections from each source as columns.
        stat_df_lists = defaultdict(list)
        for week in weeks:
            week_stats_df = self._read_weekly_data(week)
            for stat in self.stats:
                stat_df_lists[stat].append(week_stats_df[stat])
        
        self.stat_dfs = {stat: pd.concat(
            stat_df_lists[stat], ignore_index=True, sort=False) \
            for stat in self.stats}
        
    def read_data(self, stat_dfs):
        """Read stat_dfs from dict of stat dfs."""
        self.stat_dfs = stat_dfs.copy()
        
    def _read_weekly_data(self, week):
        """
        Read relevant stats for specified position and week from all
        saved datafiles and impute missing data.
        
        Parameters
        ----------
        week: int
            Week number of the season, 0 represents full season projections.
    
                 
        Returns
        -------
        stats_df: dict
            Dictionary with stat names as keys and DataFrames of
        """
        # Get all stats filenames, ignoring DFS cost files.
        fids = glob(f'../data/{self.year}/{week}/{self.pos}/*.csv')
        fids = [fid for fid in fids if not \
            any(source in fid for source in ['FanDuel', 'DraftKings'])]
        
        # Init dict of empty DataFrames for each stat.
        df = pd.DataFrame(columns=['Player', 'Team'])
        df.set_index(['Player', 'Team'], inplace=True)
        stat_dfs = {stat: df.copy() for stat in self.stats}
        
        # Read stats from each file, appending columns to each DataFrame
        # with source as column name.
        for fid in fids:
            source_df = pd.read_csv(fid)
            for stat in self.stats:
                # Drop columns other than specified stat.
                try:
                    stat_df = source_df[['Player', 'Team', stat]].copy()
                except KeyError:
                    continue  # stat not in file.

                # Rename specified stat to source name.
                source = fid.rsplit('/', 1)[1].split('.csv')[0]
                stat_df[source] = stat_df[stat]
                stat_df = stat_df[['Player', 'Team', source]]
            
                # Join to full stat DataFrame.
                stat_df.set_index(['Player', 'Team'], inplace=True)
                stat_dfs[stat] = stat_dfs[stat].join(stat_df, how='outer')

        # Clean DataFrame for each stat and impute missing data.
        for stat in self.stats:
            clean_df = self._drop_bad_indices(stat_dfs[stat].copy())
            proj_imputed_df = impute(clean_df, self.proj_impute_method)
            imputed_df = impute_realized_stats(
                proj_imputed_df, self.stat_impute_method)
            imputed_df['Week'] = week
            imputed_df.reset_index(inplace=True)
            stat_dfs[stat] = imputed_df
        
        return stat_dfs

    def _drop_bad_indices(self, df):
        """
        Drop indices with 50% or greater missing data.
        
        Parameters
        ----------
        df: pd.DataFrame
            Stat DataFrame with source columns and player/team  multi-index.
        
        Returns
        -------
        df: pd.DataFrame
            Input DataFrame with bad indices removed.
        """
        
        # If STATS file exists, ignore it when counting number of nans per row.
        if 'STATS' in list(df):
            df['STATS'].fillna('TEMP', inplace=True)
            n = len(list(df)) - 1
        else:
            n = len(list(df))
        
        # Drop rows with >= 50% missing data.
        df['NaN pct'] =  df.isnull().sum(axis=1) / n
        df = df[df['NaN pct'] <= 0.5].copy()
        df.drop('NaN pct', axis=1, inplace=True)
        
        
        # Make STATS column NaNs again.
        if 'STATS' in list(df):
            df['STATS'].replace('TEMP', np.NaN, inplace=True)
            
        return df
        

        

# %%

# self = TrainProjections(pos='QB')
# # self.load_data(weeks=range(1,18))
# # stat_dfs = self.stat_dfs
#
# self.read_data(stat_dfs)
#
# # %%
# df = self.stat_dfs['Pass Yds'].copy()
# df.head()