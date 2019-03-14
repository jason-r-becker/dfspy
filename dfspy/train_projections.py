from collections import defaultdict
from itertools import chain

import empiricalutilities as eu
import numpy as np
import pandas as pd
import statsmodels.api as sms

from project_stats import StatProjection

# %%


class TrainProjections:
    """
    Class for training ensemble methods for projections, creating
    visualizations for comparing methods, and saving model parameters.
    
    Parameters
    ----------
    pos: {'QB', 'RB', 'WR', 'TE', 'DST'}
        Player position.
    year: int, default=2018
        Year of the season.
    season: bool, default=False
        If True use full season projections, by default train on weekly data.
    data: dict[stat: pd.DataFrame], default=None
        Dictionary of all stats source projections and true realized stats
        for each projected stat.
    """
    
    def __init__(self, pos, year=2018, season=False, data=None):
        self.pos = pos
        self.season_flag = season
        self._train_pct = 0.8
        
        if self.season_flag:
            # TODO
            pass
        else:
            projector = StatProjection(pos, year, season)
            self.essential_stats = projector.essential_stats
            self.nonessential_stats = projector.nonessential_stats
            self.thresholds = projector.thresholds
            
            if data is None:
                projector.load_data(weeks=range(1, 18))
                self._stat_dfs = projector.stat_dfs
            else:
                self._stat_dfs = data.copy()
        
        self._train_dfs, self._test_dfs = self._train_test_split()
        
    def _train_test_split(self):
        """Split stat DataFrames into train/test dicts above threshold."""
        train_dfs, test_dfs = {}, {}
        for stat in self.essential_stats:
            df = self._stat_dfs[stat]
            df_no_weeks = df.drop('Week', axis=1)
            # Keep rows where average projection is above threshold.
            df = df[np.mean(df_no_weeks, axis=1) > self.thresholds[stat]].copy()
            
            if self.season_flag:
                # TODO
                pass
            else:
                train_dfs[stat] = df[df['Week'] < 15].copy()
                test_dfs[stat] = df[df['Week'] >= 15].copy()

        return train_dfs, test_dfs

    def _train_simple_linear_regression(self, stat):
        """
        Train simple linear regression model of specified stat.
        
        Parameters
        ----------
        stat: str
            Stat to train model for.
            
        Returns
        -------
        ols: sms.RegressionResultsWrapper
            OLS regression results.
            Attributes include {params, pvalues, rsquared}.
        """
        
        train_df = self._train_dfs[stat]
        # Split training portion for simple regression.
        train_df = train_df.iloc[:int(self._train_pct*len(train_df)), :]
        ignored_cols = ['Player', 'Team', 'Pos', 'POS', 'Week', 'STATS']
        cols = [col for col in list(train_df) if col not in ignored_cols]
        
        X = train_df[cols].copy()
        y = train_df['STATS'].values
        
        return sms.OLS(y, sms.add_constant(X)).fit()
        
#
# self = TrainProjections('QB')
# data = self._stat_dfs

self = TrainProjections('QB', data=data)

# %%


pos = 'QB'
stat = 'Pass Yds'



def make_simple_linear_regression_results_table():
    """Make linear regression results table for appendix."""
    positions = 'QB RB WR TE DST'.split()
    trainers = {pos: TrainProjections(pos) for pos in positions}
    
    def pval_significance(pvals):
        """Return *-style significance list for pvalues."""
        sig = []
        for pval in pvals:
            if pval <= 0.001:
                sig.append('*')
            elif pval <= 0.01:
                sig.append('**')
            elif pval <= 0.05:
                sig.append('***')
            else:
                sig.append('')
        return sig
    
    # Get OLS results for each essential stat.
    df = pd.DataFrame()
    for pos in positions:
        for stat in trainers[pos].essential_stats:
            ols = trainers[pos]._train_simple_linear_regression(stat)
            # Find signifcance and rounded values.
            vals = [pos, stat] + [f'{val}{sig}' for val, sig in \
                    zip(ols.params.round(2), pval_significance(ols.pvalues))]
            cols = ['Position', 'Stat'] + list(ols.params.index)
            temp_df = pd.DataFrame([vals], columns=cols)
            df = df.append(temp_df, sort=False)
            
    
    # Change positon column to only indicate position once.
    n_pos = {pos: len(trainers[pos].essential_stats)-1 for pos in positions}
    pos_list = list(chain(*[[pos] + n_pos[pos]*[''] for pos in positions]))
    df['Position'] = pos_list
    
    # Build and print table.
    col_fmt = 'l' * (len(list(df)) + 1)
    cap = 'Simple linear regression results for ensemble weighting of each '
    cap += 'source. Asterisks denote statistical significance of regression '
    cap += 'coefficients, (*) denoting a p-value less than 0.5, (**) for less '
    cap += 'than 0.01, and (***) for less than 0.001.'
    eu.latex_print(df, hide_index=True, adjust=True, col_fmt=col_fmt,
                   caption=cap)
    
