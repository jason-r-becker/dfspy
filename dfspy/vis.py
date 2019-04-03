import warnings
from collections import defaultdict
from glob import glob

import empiricalutilities as eu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from train_projections import TrainProjections
from project_stats import StatProjection
from scoring import get_score

plt.style.use('fivethirtyeight')
# %matplotlib qt
# %%

def main():
    fdir = '../figures'
    save = False
    # plot_missing_data(fdir, save)
    # plot_stat_hists(fdir, save)
    
# %%
def plot_missing_data(fdir='../figures', save=True):
    """Plot # of sources and % data missing for each essential stat."""
    stats = {
        'QB': ['Pass Yds', 'Pass TD', 'Pass Int', 'Rush Yds', 'Rush TD'],
        'RB': ['Rush Yds', 'Rush TD', 'Receptions', 'Rec Yds', 'Rec TD'],
        'WR': ['Rush Yds', 'Rush TD', 'Receptions', 'Rec Yds', 'Rec TD'],
        'TE': ['Receptions', 'Rec Yds', 'Rec TD'],
        'DST': ['PA', 'YdA', 'TD', 'Sack', 'Int', 'Fum Rec'],
        }
    n_stats = sum(len(s) for s in stats.values())
    
    n_sources = np.zeros(n_stats)
    pct_nan = np.zeros(n_stats)
    x_ticks = []
    i = 0
    for pos in stats.keys():
        proj = TrainProjections(pos)
        proj.load_data(weeks=range(1,18), impute_method=False)
        for stat in stats[pos]:
            df = proj._stats_df[stat]
            
            # Subset DataFrame to only include only projection columns.
            ignored_cols = ['Player', 'Team', 'Pos', 'Week', 'STATS']
            keep_cols = [c for c in list(df) if c not in ignored_cols]
            proj_df = df[keep_cols].copy()
            
            # Find percentage of NaNs in for the stat and # of sources.
            n, m = proj_df.shape
            pct_nan[i] = np.sum(np.sum(proj_df.isnull())) / (n * m)
            n_sources[i] = m
            x_ticks.append(f'{pos} - {stat}')
            i += 1
    
    # Creat figure.
    fig, ax = plt.subplots(1, 1, figsize=[12, 6])
    x = np.arange(n_stats)
    ax.bar(x, n_sources, color='steelblue', alpha=0.7)
    ax.bar(x, n_sources*pct_nan, color='firebrick', alpha=0.9)
    for xt, yt, s in zip(x, n_sources*pct_nan, pct_nan):
        ax.text(xt, yt+0.05, f'{s:.1%}', color='firebrick', ha='center',
                fontsize=7)
    ax.set_ylabel('Number of Sources')
    ax.set_xlabel('Projected Stat')
    plt.xticks(x, x_ticks, rotation=45, ha='right')
    ax.xaxis.grid(False)
    
    # Save figure.
    if save:
        fid = 'missing_data'
        eu.save_fig(fid, dir=fdir)
        cap = 'Number of sources collected for each essential stat (blue). Red '
        cap += 'indicates percentage of missing data for each respective stat.'
        eu.latex_figure(fid, dir=fdir, width=0.95, caption=cap)

    
    
    # Nonessential stats.
    stats = {
        'QB': ['Receptions', 'Rec Yds', 'Rec TD', '2PT'],
        'RB': ['Pass Yds', 'Pass TD', 'Pass Int', '2PT'],
        'WR': ['Pass Yds', 'Pass TD', 'Pass Int', '2PT'],
        'TE': ['Pass Yds', 'Pass TD', 'Pass Int', 'Rush Yds', 'Rush TD', '2PT'],
        'DST': ['Saf', 'Blk'],
        }
    n_stats = sum(len(s) for s in stats.values())
    
    n_sources = np.zeros(n_stats)
    pct_nan = np.zeros(n_stats)
    x_ticks = []
    i = 0
    for pos in stats.keys():
        proj = TrainProjections(pos)
        proj.load_data(weeks=range(1,18), impute_method=False)
        for stat in stats[pos]:
            df = proj._stats_df[stat]
            
            # Subset DataFrame to only include only projection columns.
            ignored_cols = ['Player', 'Team', 'Pos', 'Week', 'STATS']
            keep_cols = [c for c in list(df) if c not in ignored_cols]
            proj_df = df[keep_cols].copy()
            
            # Find percentage of NaNs in for the stat and # of sources.
            n, m = proj_df.shape
            pct_nan[i] = np.sum(np.sum(proj_df.isnull())) / (n * m)
            n_sources[i] = m
            x_ticks.append(f'{pos} - {stat}')
            i += 1
    
    # Creat figure.
    fig, ax = plt.subplots(1, 1, figsize=[12, 6])
    x = np.arange(n_stats)
    ax.bar(x, n_sources, color='steelblue', alpha=0.7)
    ax.bar(x, n_sources*pct_nan, color='firebrick', alpha=0.9)
    for xt, yt, s in zip(x, n_sources*pct_nan, pct_nan):
        ax.text(xt, yt+0.05, f'{s:.1%}', color='firebrick', ha='center',
                fontsize=7)
    ax.set_ylabel('Number of Sources')
    ax.set_xlabel('Projected Stat')
    plt.yticks(np.arange(max(n_sources)+1))
    plt.xticks(x, x_ticks, rotation=45, ha='right')
    ax.xaxis.grid(False)
    
    # Save figure.
    if save:
        fid = 'nonessential_missing_data'
        eu.save_fig(fid, dir=fdir)
        cap = 'Number of sources collected for each nonessential stat (blue). '
        cap += 'Red  indicates percentage of missing data for each '
        cap += 'respective stat.'
        eu.latex_figure(fid, dir=fdir, width=0.95, caption=cap)
        

def plot_stat_hists(fdir='../figures', save=True):
    """
    Plot all essential stat histograms for the appendix and
    an example histograms figure for the main body of the paper.
    """
    # Load data.
    positions = 'QB RB WR TE DST'.split()
    stats = {pos: TrainProjections(pos) for pos in positions}
    for pos in positions:
        stats[pos].load_data(weeks=range(1, 18))
    
    
    # Plot all no threshold histograms for appendix.
    no_thresh_fids = {pos: f'no_threshold_hist_{pos}' for pos in positions}
    for pos in positions:
        n = len(stats[pos].essential_stats)
        fig, axes = plt.subplots(n, 1, figsize=[6, 2.5*n])
        for stat, ax in zip(stats[pos].essential_stats, axes.flat):
            stats[pos].plot_projection_hist(
                stat, bins=None, threshold=False, ax=ax)
        plt.tight_layout()
        if save:
            eu.save_fig(no_thresh_fids[pos], dir=fdir)
        else:
            plt.show()
            
    # Plot all threshold histograms for appendix.
    thresh_fids = {pos: f'threshold_hist_{pos}' for pos in positions}
    for pos in positions:
        n = len(stats[pos].essential_stats)
        fig, axes = plt.subplots(n, 1, figsize=[6, 2.5*n])
        for stat, ax in zip(stats[pos].essential_stats, axes.flat):
            stats[pos].plot_projection_hist(
                stat, bins=None, threshold=True, ax=ax)
        plt.tight_layout()
        if save:
            eu.save_fig(thresh_fids[pos], dir=fdir)
        else:
            plt.show()
    
    # Create LaTeX code to plot figures.
    if save:
        for pos in positions:
            cap = 'Essential stat raw histograms and thresholded histograms '
            cap += f'for {pos}.'
            
            eu.latex_figure(
                fids=[no_thresh_fids[pos], thresh_fids[pos]],
                dir=fdir,
                subcaptions=['Raw histogram with threshold (red).',
                             'Histogram above threshold.'],
                caption=cap,
                width=0.9,
                )
            print()
            print('\pagebreak')
    
    # Plot raw histograms for example stats.
    fids = ['no_theshold_example_hists', 'no_theshold_example_hists_RB']
    fig, axes = plt.subplots(2, 1, figsize=[6, 9])
    for pos, stat, ax in zip(['QB', 'RB'], ['Pass Yds', 'Rush Yds'], axes.flat):
        stats[pos].plot_projection_hist(
                stat, bins=None, threshold=False, ax=ax)
    plt.tight_layout()
    if save:
        eu.save_fig(fids[0], dir=fdir)
    else:
        plt.show()
        
    # Plot thresholded histograms for example stats.
    fids = ['no_theshold_example_hists', 'no_theshold_example_hists_RB']
    fig, axes = plt.subplots(2, 1, figsize=[6, 9])
    for pos, stat, ax in zip(['QB', 'RB'], ['Pass Yds', 'Rush Yds'], axes.flat):
        stats[pos].plot_projection_hist(
            stat, bins=None, threshold=True, ax=ax)
    plt.tight_layout()
    if save:
        eu.save_fig(fids[1], dir=fdir)
    else:
        plt.show()
            
    # Create LaTeX code to plot example stats figure.
    if save:
        print('\n\n\n')
        cap = 'Example raw and thresholded histograms for QB passing yards '
        cap += f'and RB rushing yards.'
        eu.latex_figure(
            fids=fids,
            dir=fdir,
            subcaptions=['Raw histogram with threshold (red).',
                         'Histogram above threshold.'],
            caption=cap,
            width=0.9,
            )
                

def plot_weekly_pos_projections(pos, week, year=2018, n_players=None,
    league='FanDuel', ax=None, figsize=[12, 12]):
    """
    Plot weekly projections for specified position.
    
    Parameters
    ----------
    pos: {'QB', 'RB', 'WR', 'TE', 'DST'}
        Position to plot.
    week: int
        Week of the season to plot.
    year: int, default=2018
        Year of season to plot.
    n_players: int, default=None
        Number of players to include in plot.
        If None, the respective numbers are used for each positon:
        {'QB': 30, 'RB': 50, 'WR': 50, 'TE': 30, 'DST': 20}.
    league: {'FanDuel', 'DraftKings', 'ESPN', 'Yahoo'}
        League name to apply scoring rules from.
    ax: matplotlib axis, default=None
        Matplotlib axis to plot figure, if None one is created.
    figsize: list or tuple, default=(6, 6)
        Figure size.
    """

    n_players = {'QB': 30, 'RB': 50, 'WR': 50, 'TE': 30, 'DST': 20}[pos] \
        if n_players is None else n_players
    
    # Load floor, projection, ceiling, and status for each player.
    path = f'../data/{year}/{week}/{pos}/proj/{{}}.csv'
    def get_scores(score_type, path=path, pos=pos, league=league):
        """Load data for FLOOR, CEIL, or PROJ and apply scoring rules."""
        df = pd.read_csv(path.format(score_type))
        df.set_index(['Player', 'Team'], inplace=True)
        df[score_type] = get_score(df, pos, league)
        return pd.DataFrame(df[score_type])
    
    # Init empty DataFrame with proper index.
    df = pd.DataFrame(columns=['Player', 'Team'])
    df.set_index(['Player', 'Team'], inplace=True)
    for score_type in 'FLOOR PROJ CEIL'.split():
        df = df.join(get_scores(score_type), how='outer')
    
    # Load status of players.
    espn_df = pd.read_csv(f'../data/{year}/{week}/{pos}/ESPN.csv')
    espn_df.set_index(['Player', 'Team'], inplace=True)
    df = df.join(pd.DataFrame(espn_df['Status']), how='inner')
    df['Status'].fillna(' ', inplace=True)
    
    # Sort players by projected points and determine ranking.
    df = df[df['FLOOR'] != df['CEIL']].copy()
    df.sort_values('PROJ', inplace=True, ascending=False)
    df = df.iloc[:n_players, :].copy()
    df.sort_values('PROJ', inplace=True, ascending=True)
    df.reset_index(inplace=True)
    df['RANK'] = n_players - df.index.values
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    xerr = [(df['PROJ']- df['FLOOR']).values,
            (df['CEIL']- df['PROJ']).values]
    
    ax.errorbar(df['PROJ'], df['RANK'], xerr=xerr, fmt='o', ms=4,
                color='k', capsize=3, markeredgewidth=0.5, lw=0.5)
    for _, row in df.iterrows():
        if row['Status'] != ' ':
            ax.text(row['CEIL']+0.2, row['RANK'], row['Status'], fontsize=6,
                    color='firebrick', ha='left', va='center')
            offset = 0.5
        else:
            offset = 0.2
        ax.text(row['CEIL']+offset, row['RANK'], row['Player'], fontsize=6,
                ha='left', va='center')
        
    ax.set_ylabel('Rank')
    ax.set_xlabel('Projected Points')
    plt.ylim(n_players+2, -2)
    plt.tight_layout()
    
# %%
for i in range(1, 6):
    plot_weekly_pos_projections('DST', i)
    plt.show()


# %%
class CompareModels:
    """
    Load data for model comparison and compare MAE values across all
    models for given stat.
    
    Parameters
    ----------
    stats: {'essential', 'essential_below_thresh', 'non_essential'}
        Stats to compare.
    period: {'weekly', 'season'}
        Season or weekly models.
    subset: {'train', 'cv', 'test'}
        Train/cv/test set comparison.
    year: int, default=2018
        Year to collect results.
        
        
    Methods
    -------
    plot_color_table(): Plot colorized table of MAE values for each algorithm.
    save_optimal_projections(): Save optimal algorithm for each positon-stat
        pair to a .json file to be accessed by `project_stats.py`.
    """
    
    def __init__(self, stats, period, subset, year=2018):
        
        self.stats = stats
        self.period = period
        self.subset = subset
        self.year = year
        
        if period == 'weekly':
            self.positions = 'QB RB WR TE DST'.split()
        else:
            self.positions = 'QB RB WR TE K DST'.split()
        
        self.all_stat_dfs = self._load_data()
        self.mae_df = self._build_mae_df()
        
    def _read_weekly_data(self, pos, week):
        """
        Read relevant stats for specified position and week from all
        saved datafiles and impute missing data.
        
        Parameters
        ----------
        pos: {'QB', 'RB', 'WR', 'TE', 'K', 'DST'}
            Player position.
        week: int
            Week number of the season, 0 represents full season projections.
    
                 
        Returns
        -------
        stats_df: dict
            Dictionary with stat names as keys and DataFrames of
        """
        # Get all stats filenames, ignoring DFS cost files.
        stat_fid = f'../data/{self.year}/{week}/{pos}/STATS.csv'
        if self.stats == 'essential':
            pass
        elif self.stats == 'essential_below_thresh':
            sources = ['CEIL', 'FLOOR', 'MEAN', 'MEDIAN']
        elif self.stats == 'non_essential':
            sources = ['CEIL', 'FLOOR', 'MEAN', 'MEDIAN']
        
        fids = glob(f'../data/{self.year}/{week}/{pos}/proj/*.csv')
        if self.stats != 'essential':
            # Keep only non ML sources.
            fids = [fid for fid in fids \
                    if any(source in fid for source in sources)]
        else:
            fids = [fid for fid in fids if 'PROJ' not in fid]
        fids.append(stat_fid)
        
        # Load correct stats and thresholds.
        proj = StatProjection(pos, self.year, self.period=='season')
        thresholds = proj.thresholds
        if self.stats == 'non_essential':
            stats = proj.nonessential_stats
        else:
            stats = proj.essential_stats
        
        # Init dict of empty DataFrames for each stat.
        df = pd.DataFrame(columns=['Player', 'Team'])
        df.set_index(['Player', 'Team'], inplace=True)
        stat_dfs = {stat: df.copy() for stat in stats}
        
        # Read stats from each file, appending columns to each DataFrame
        # with source as column name.
        for fid in fids:
            source_df = pd.read_csv(fid)
            for stat in stats:
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
        for stat in stats:
            clean_df = stat_dfs[stat].copy().fillna(0)
            clean_df['Week'] = week
            clean_df.reset_index(inplace=True)
            if self.stats == 'essential':
                clean_df = clean_df.loc[clean_df['STATS'] > thresholds[stat]]
            elif self.stats == 'non_essential':
                pass
            else:
                clean_df = clean_df.loc[clean_df['STATS'] <= thresholds[stat]]
                
            # Drop duplicate player indexes, keeping only highest projected.
            players = list(clean_df['Player'])
            dupes = list(set([x for x in players if players.count(x) > 1]))
            ix = [x not in dupes for x in players]
            final_df = clean_df[ix].copy()
            
            for player in dupes:
                dupe_df = np.max(
                    clean_df[clean_df['Player'] == player].copy(), axis=0)
                final_df = final_df.append(dupe_df, ignore_index=True)
            
            # Append to stat DataFrames dict.
            final_df.fillna(0, inplace=True)
            stat_dfs[stat] = final_df
        
        return stat_dfs
    
    def _load_data(self):
        """
        Load data for train/cv/test subset of all projections.
        
        Returns
        -------
        all_stat_dfs: dict[pos: {stat: pd.DataFrame}]
            Dict with positions keys, followed by relevant stat keys with
            DataFrame values for all projections.
        """
        if self.subset == 'train':
            weeks = range(1, 11)
        elif self.subset == 'cv':
            weeks = range(11, 15)
        elif self.subset == 'test':
            weeks = range(15, 18)
        
        # Make dictionary of
        all_stat_dfs = {}
        for pos in self.positions:
            stat_df_lists = defaultdict(list)
            for week in weeks:
                week_stats_dfs = self._read_weekly_data(pos, week)
                for stat in week_stats_dfs.keys():
                    df = week_stats_dfs[stat]
                    bad_cols = 'Player Team Week'.split()
                    cols = [c for c in df.columns \
                            if not any(c in bad_cols for bc in bad_cols)]
                    stat_df_lists[stat].append(df[cols].copy())
        
            stat_dfs = {stat: pd.concat(
                stat_df_lists[stat], ignore_index=True, sort=False) \
                for stat in week_stats_dfs.keys()}
            
            for stat, df in stat_dfs.items():
                stat_dfs[stat] = df.fillna(0)
                
            all_stat_dfs[pos] = stat_dfs
        
        
        return all_stat_dfs
        
    def _compute_MAE(self, df):
        """Compute MAE for each method of a stat DataFrame.
        
        Returns
        -------
        mae: dict[str: float]
            Dict of method names and respective mae values.
        """
        
        if 'STATS' not in list(df.columns):
            df['STATS'] = 0
        cols = [c for c in df.columns if c != 'STATS']
        mae = {c: np.mean(np.abs(df[c] - df['STATS']).values) for c in cols}
        return mae
        
    def _build_mae_df(self):
        """
        Make DataFrame of MAE results for all relevant stats.
        
        Returns
        mae_df: pd.DataFrame
            DataFrame of MAE values for all stats and projection methods.
        """
        mae_d = defaultdict(list)
        for pos in self.positions:
            for stat, df in self.all_stat_dfs[pos].items():
                mae_vals = self._compute_MAE(df)
                mae_d['Position'].append(pos)
                mae_d['Stat'].append(stat)
                for method, mae in mae_vals.items():
                    mae_d[method].append(mae)
        return pd.DataFrame(mae_d)
    
    def plot_color_table(self, ax=None, figsize=(6, 6), fontsize=6, prec=1):
        """
        Plot asset colorize table of MAE values.

        Parameters
        ----------
        ax: matplotlib axis, default=None
            Matplotlib axis to plot figure, if None one is created.
        figsize: list or tuple, default=(6, 6)
            Figure size.
        fontsize: int, default=6
            Fontsize for stat and method labels.
        prec: int, default=2
            Format precision for MAE values in table.
        """

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            
        # Clean DataFrame and save MAE values as labels.
        df = self.mae_df.copy()
        fmt_methods = {
            'MEAN': 'Mean',
            'MEDIAN': 'Median',
            'FLOOR': 'Min',
            'CEIL': 'Max',
            }
        # Sort and format column names.
        df.columns = [fmt_methods.get(c, c) for c in df.columns]
        df = df[eu.custom_sort(df.columns, 'Mia')]
        
        # Update index to pos-stat str.
        stats = [f'{pos} - {stat}' for pos, stat in \
            zip(df['Position'].values, df['Stat'].values)]
        df.index = stats
        df.drop(['Position', 'Stat'], inplace=True, axis=1)
        labels = df.values
        
        # Find color values from normalized data.
        cdf = df.copy()
        cdf = cdf.transpose()
        cdf -= cdf.mean(axis=0)
        cdf /= cdf.std(axis=0)
        cdf = cdf.transpose()
        
        # Make heatmap plot.
        sns.heatmap(cdf, cmap='RdYlGn_r', linewidths=0.5, annot=labels,
                    fmt=f'0.{prec}f', cbar=False, ax=ax)
        
        ax.xaxis.set_ticks_position('top')
        ax.set_xticklabels(df.columns, fontsize=fontsize, format='bf')
        ax.set_yticklabels(df.index, fontsize=fontsize)
        
    def save_optimal_projections(self):
        pass
    

# %%


def make_algo_comparison_tables(year=2018, fdir='../figures', save=True):
    # %%
    ess_train = CompareModels('essential', period='weekly', subset='train')
    ess_cv = CompareModels(stats='essential', period='weekly', subset='cv')
    ess_thresh = CompareModels(
        'essential_below_thresh',period='weekly', subset='train')
    non_ess = CompareModels('non_essential', period='weekly', subset='train')
    
    # %%
    # Essential stats.
    
    
    fig, axes=plt.subplots(1, 2, figsize=(18, 12))
    ess_train.plot_color_table(ax=axes[0], fontsize=12, prec=2)
    ess_cv.plot_color_table(ax=axes[1], fontsize=12, prec=2)
    plt.tight_layout()
    if save:
        eu.save_fig('essential_MAE_color_table', dir=fdir)
    else:
        plt.show()
    
    ess_train.plot_color_table(figsize=(9, 12), fontsize=12)
    if save:
        eu.save_fig('essential_train_MAE_table', dir=fdir)
    else:
        plt.show()
    
    
    ess_cv.plot_color_table(figsize=(9, 12), fontsize=12)
    if save:
        eu.save_fig('essential_cv_MAE_table', dir=fdir)
        cap = 'Mean Absolute Error (MAE) values for all essential stats '\
            'and studied algorithms.'
        eu.latex_figure(
            fids=['essential_train_MAE_table', 'essential_cv_MAE_table'],
            dir=fdir,
            caption=cap,
            subcaptions=['Training MAE.', 'Cross-Validation MAE.'],
            width=0.98,
            )
    else:
        plt.show()
    # %%
