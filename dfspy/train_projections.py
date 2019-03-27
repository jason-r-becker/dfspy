import argparse
import os
import warnings
from collections import defaultdict
from glob import glob
from itertools import chain
from time import perf_counter

import empiricalutilities as eu
import joblib
import numpy as np
import pandas as pd
import pickle
import statsmodels.api as sms
from operator import itemgetter
from scipy.stats import beta, expon, randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost.sklearn import XGBRegressor

from project_stats import StatProjection

# %%

def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    make_model_dirs()

    # Read pickled data if it exists, otherwise generate imputed data.
    positions = 'QB RB WR TE DST'.split()
    try:
        with open('../data/.imputed_data', 'rb') as fid:
            data = pickle.load(fid)
    except FileNotFoundError:
        trainer_dict = {pos: TrainProjections(pos) for pos in positions}
        data = {pos: trainer_dict[pos]._stat_dfs for pos in positions}
        # Store imputed data.
        with open('../data/.imputed_data', 'wb') as fid:
            pickle.dump(data, fid)
    trainer_dict = {
        pos: TrainProjections(pos, data=data[pos]) for pos in positions}
    
    # Train model on essential stats.
    args = parse_args()
    if args.model == 'WEIGHTED':
        train_simple_linear_regression(
            trainers=trainer_dict,
            period=args.period,
            save=args.save,
            verbose=args.verbose,
            )
    else:
        train_ml_models(
            model=args.model,
            trainers=trainer_dict,
            n_iters=args.n_iters,
            period=args.period,
            save=args.save,
            verbose=args.verbose,
            )


def parse_args():
    """Collect settings from command line and set defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Model Name')
    parser.add_argument('-p', '--period', help='weekly/season')
    parser.add_argument(
        '-n', '--n_iters', type=int, help='Number of training iterations')
    parser.add_argument('-s', '--save', action='store_true', help='Save Models')
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='Verbosity')
    parser.set_defaults(model='LR', period='weekly', n_iters=500)
    return parser.parse_args()

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
    
    Methods
    -------
    train_simple_linear_regression(): train WEIGHTED simple linear
        regression model.
    
    """
    
    def __init__(self, pos, year=2018, season=False, data=None):
        self.pos = pos
        self.season_flag = season
        self._train_pct = 0.8
        
        if self.season_flag:
            self.period = 'season'
            # TODO
            pass
        else:
            self.period = 'weekly'
            projector = StatProjection(pos, year, season)
            self.essential_stats = projector.essential_stats
            self.nonessential_stats = projector.nonessential_stats
            self.thresholds = projector.thresholds
            
            if data is None:
                projector.load_data(weeks=range(1, 18))
                self._stat_dfs = projector.stat_dfs
            else:
                self._stat_dfs = data.copy()
        
        self._train_dfs, self._cv_dfs, self._test_dfs = self._train_test_split()
        
    def _train_test_split(self):
        """Split stat DataFrames into train/cv/test dicts above threshold."""
        train_dfs, cv_dfs, test_dfs = {}, {}, {}
        for stat in self.essential_stats:
            df = self._stat_dfs[stat]
            df_no_weeks = df.drop('Week', axis=1)
            # Keep rows where average projection is above threshold.
            df = df[np.mean(df_no_weeks, axis=1) > self.thresholds[stat]].copy()
            
            if self.season_flag:
                # TODO
                pass
            else:
                train_dfs[stat] = df[df['Week'] < 12].copy()
                cv_dfs[stat] = df[(df['Week'] >= 12) & (df['Week'] < 15)].copy()
                test_dfs[stat] = df[df['Week'] >= 15].copy()

        return train_dfs, cv_dfs, test_dfs

    def train_simple_linear_regression(self, stat):
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
        ignored_cols = ['Player', 'Team', 'Pos', 'POS', 'Week', 'STATS']
        cols = [col for col in sorted(train_df) if col not in ignored_cols]
        
        X = train_df[cols].copy()
        y = train_df['STATS'].values
        
        return sms.OLS(y, sms.add_constant(X)).fit()

    def train_machine_learning_algorithm(self, model, stat, n_iters, kfolds=5):
        """
        Train simple linear regression model of specified stat.
        
        Parameters
        ----------
        stat: str
            Stat to train model for.
        model: {'LR', 'SVR', 'RF', 'XGB'}
            Machine learning algorithm to train.
                - LR: Projection stats are computed using advanced linear
                      regression with regularization.
                - SVR: Projection stats are computed with
                       support vector machines.
                - RF: Projection stats are computed using a random forest.
                - XGB: Projection stats are computed using eXtreme gradient
                       boosted trees.
        n_iters: int
            Number of training iterations.
        kfolds: int, default=5
            Number of kfolds to use for cross-validation.
            
        Returns
        -------
        final_model: sklearn model class
            Best model from cross-validation training.
        cv_res: dict
            Cross-validation results dict.
        """
        # Get X and y data for machine learning models.
        train_df = self._train_dfs[stat]
        ignored_cols = ['Player', 'Team', 'Pos', 'POS', 'Week', 'STATS']
        cols = [col for col in sorted(train_df) if col not in ignored_cols]
        
        X = train_df[cols].values
        y = train_df['STATS'].values
        
        # Standardize data.
        # Load scaler if it has been saved, else scale and save.
        fid = f"../data/.models/{self.period}/{self.pos}/" \
            f"{stat.replace(' ', '_')}/scaler.sav"
        try:
            scaler = joblib.load(fid)
        except FileNotFoundError:
            scaler = StandardScaler().fit(X)
            joblib.dump(scaler, fid)
        
        X_scaled = scaler.transform(X)
        
        # Make stratified kfolds from deciles of stat values.
        y_bins = pd.qcut(y, 10, labels=False, duplicates='drop')
        y_bins = np.array(
            pd.qcut(pd.Series(y).rank(method='first'), 10, labels=False))
        skfolds = StratifiedKFold(
            n_splits=kfolds, shuffle=True, random_state=8675309)
        
        # Set up parameters for CV search.
        params = {
            'LR': {'alpha': expon(), 'l1_ratio': uniform()},
            'SVR': {'kernel': ['rbf'], 'C': expon(), 'gamma': expon()},
            'RF': {
                'n_estimators': randint(low=1, high=100),
                'max_depth': randint(low=2, high=6),
                },
            'XGB': {
                'n_estimators': randint(3, 40),
                'max_depth': randint(3, 40),
                'learning_rate': uniform(0.05, 0.4),
                'colsample_bytree': beta(10, 1) ,
                'subsample': beta(10, 1) ,
                'gamma': uniform(0, 10),
                'reg_alpha': expon(0, 50),
                'min_child_weight': expon(0, 50),
                },
            }[model]
            
        mod = {
            'LR': ElasticNet,
            'SVR': SVR,
            'RF': RandomForestRegressor,
            'XGB': XGBRegressor,
            }[model]
        
        
        
        # Train model with cross-validation.
        mod_kwargs = {'n_threads': -1} if model == 'XGB' else {}
        n_jobs = 1 if model == 'XGB' else -1
        rnd_search = RandomizedSearchCV(
            mod(mod_kwargs),
            param_distributions=params,
            n_iter=n_iters,
            cv=skfolds.split(X_scaled, y_bins),
            scoring='neg_mean_absolute_error',
            n_jobs=n_jobs,
            )
        
        rnd_search.fit(X_scaled, y)
        final_model = rnd_search.best_estimator_
        cv_res = rnd_search.cv_results_

        return final_model, cv_res
        
        
# pos = 'QB'
# stat = 'Pass Yds'
# self = TrainProjections(pos, data=data[pos])
# model = 'RF'
# model = 'SVR'
# %%

def mkdir(directory):
    """Make directory if it does not already exist."""
    try:
        os.makedirs(directory)
    except OSError:
        pass


def make_model_dirs():
    positions = 'QB RB WR TE DST'.split()
    for pos in positions:
        projector = StatProjection(pos)
        stats = projector.essential_stats + projector.nonessential_stats
        for stat in stats:
            mkdir(f"../data/.models/weekly/{pos}/{stat.replace(' ', '_')}")
            mkdir(f"../data/.models/season/{pos}/{stat.replace(' ', '_')}")
        
        
def train_simple_linear_regression(
    trainers, period, n_iters, save, verbose):
    """
    Train linear regression weighted model.
    
    Parameters
    ----------
    trainers: dict
        Dict with postions as keys and TrainProjections class as values.
    period: {'weekly', 'season'}
        Period to train models over.
    n_iters: int
        Number of iterations to train models.
    save: bool
        If True, save results into data/.models dir.
    verbose: bool
        If True, print progress bar to screen.
    """
    
    if verbose:
        print('\n\nSimple linear regresssion WEIGHTED model:')
        t0 = perf_counter()
    positions = trainers.keys()
    
    def fmt_pval(pvals):
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
    dis = not verbose
    n = sum(len(trainer.essential_stats) for trainer in trainers.values())
    with tqdm(total=n, disable=dis) as pbar:
        for pos in positions:
            for stat in trainers[pos].essential_stats:
                ols = trainers[pos].train_simple_linear_regression(stat)
                # Find signifcance and rounded values.
                vals = [pos, stat] + [f'{val}{sig}' for val, sig in \
                        zip(ols.params.round(2), fmt_pval(ols.pvalues))]
                cols = ['Position', 'Stat'] + list(ols.params.index)
                temp_df = pd.DataFrame([vals], columns=cols)
                df = df.append(temp_df, sort=False)
                
                if save:
                    fid = f'../data/.models/{period}/{pos}/' \
                        f'{stat.replace(" ", "_")}/WEIGHTED.csv'
                        
                    save_df = pd.DataFrame(ols.params, columns=['vals'])
                    save_df.to_csv(fid)
                    
                pbar.update(1)
                
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
    if verbose:
        print(f'Time to train: {(perf_counter() - t0) / 60:.1f} min.')
        eu.latex_print(
            df, hide_index=True, adjust=True, col_fmt=col_fmt, caption=cap)
    
        
def train_ml_models(
    trainers, model, n_iters, period, save, verbose):
    """
    Train linear regression weighted model.
    
    Parameters
    ----------
    trainers: dict
        Dict with postions as keys and TrainProjections class as values.
    model: {'LR', 'SVR', 'RF'}
        Machine learning algorithm to train.
            - LR: Projection stats are computed using advanced linear
                  regression with regularization.
            - SVR: Projection stats are computed with
                   support vector machines.
            - RF: Projection stats are computed using a random forest.
    period: {'weekly', 'season'}
        Period to train models over.
    save: bool
        If True, save results into data/.models dir.
    verbose: bool
        If True, print progress bar to screen.
    """

    positions = trainers.keys()
    n = sum(len(trainer.essential_stats) for trainer in trainers.values())
    with tqdm(total=n, disable=(not verbose)) as pbar:
        for pos in positions:
            for stat in trainers[pos].essential_stats:
                if verbose:
                    tqdm.write(f'')
                    t0 = perf_counter()
                    
                mod, res = trainers[pos].train_machine_learning_algorithm(
                    model, stat, n_iters)
                
                if save:
                    fid = f'../data/.models/{period}/{pos}/' \
                        f'{stat.replace(" ", "_")}/{model}.sav'
                    joblib.dump(mod, fid)
                    
                if verbose:
                    msg = f'{model} for {pos}-{stat}: ' \
                        f'\tTime: {perf_counter() - t0:.1f} sec' \
                        f"\tMAE: {-np.max(res['mean_test_score']):0.2f}"
                    tqdm.write(msg)
                pbar.update(1)


if __name__ == '__main__':
    main()
# %%
