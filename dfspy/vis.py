import warnings

import empiricalutilities as eu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from project_stats import TrainProjections
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
        path = f'../data/{year}/{week}/{pos}/{{}}.csv'
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
        espn_df = pd.read_csv(path.format('ESPN'))
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
        plt.show()
        

# plot_weekly_pos_projections('RB', week=6)
