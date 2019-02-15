import os
import numpy as np
import cvxpy as cp
import pandas as pd
from tqdm import tqdm
from scoring import *
from scrape_data import mkdir
# %%

class LineupOptimizer:
    def __init__(self, year, week, league):
        self.year = year
        self.week = week
        self.league = league
        self.budget = {
            'FanDuel': 60000,
            'DraftKings': 50000,
            'ESPN': None,
            }[league]
        self.data = self._load_data()

    def _load_data(self, source='NFL'):
        POS = 'QB RB WR TE DST'.split()
        d = {
            'QB': scoring_QB,
            'RB': scoring_RB,
            'WR': scoring_WR,
            'TE': scoring_TE,
            'DST': scoring_DST,
            }

        player_dfs = {}
        for pos in POS:
            filepath = f'../data/{self.year}/{self.week}/{pos}/'
            df = pd.read_csv(filepath+source+'.csv')
            df = d[pos](df)
            costs = pd.read_csv(filepath+self.league+'.csv')
            costs.columns = 'player team salary'.split()
            if pos == 'DST':
                df = df.set_index('team').join(
                    costs.set_index('team'),
                    how='left', sort=False, rsuffix='_').dropna()
                df = df.drop('player_', axis=1)
                df.reset_index(inplace=True)
                df = df['player team pos proj salary'.split()]
            else:
                df = df.set_index(['player', 'team']).join(
                    costs.set_index(['player', 'team']),
                    how='left', sort=False).dropna()
                df.reset_index(inplace=True)

            player_dfs[pos] = df

        df = pd.concat(player_dfs).reset_index(drop=True)
        df = df.join(pd.get_dummies(df['pos']), how='left')
        df = df.join(pd.get_dummies(df['team']), how='left')
        df = df.sort_values(['pos', 'team']).reset_index(drop=True)
        return df

    def _data_col_vector(self, col):
        """Return N by 1 vector of data column from self.data """
        return self.data[col].values.reshape(-1, 1)

    def _optimize(self, pt_lim):
        N = len(self.data)
        W = cp.Variable((N, 1), boolean=True)
        idx = self.data[self.data['pos']=='DST'].index.max()+1

        teams = self.data['team'].unique()
        constrs = [
            cp.matmul(W.T, self._data_col_vector('salary'))<=self.budget,
            cp.matmul(W.T, self._data_col_vector('proj'))<=pt_lim,
            cp.sum(W)==9,
            cp.matmul(W.T, self._data_col_vector('QB')) == 1,
            cp.matmul(W.T, self._data_col_vector('RB')) <= 3,
            cp.matmul(W.T, self._data_col_vector('RB')) >= 2,
            cp.matmul(W.T, self._data_col_vector('WR')) <= 4,
            cp.matmul(W.T, self._data_col_vector('WR')) >= 3,
            cp.matmul(W.T, self._data_col_vector('TE')) <= 2,
            cp.matmul(W.T, self._data_col_vector('TE')) >= 1,
            #cp.matmul(W.T, self._data_col_vector('K'))==1,
            cp.matmul(W.T, self._data_col_vector('DST')) == 1,
            cp.max(cp.matmul(W[idx:].T,
                self.data[teams].values[idx:, :])) <= self.max_players_per_team,
            ]

        obj = cp.Maximize(cp.matmul(W.T, self._data_col_vector('proj')))
        prob = cp.Problem(obj, constrs)
        prob.solve()
        W.value = W.value.round()
        idx = []
        for i, w in enumerate(W.value):
            if w == 1:
                idx.append(i)
        proj_pts = self.data.iloc[idx]['proj'].sum()
        lineup = self.data.iloc[idx]['player team pos proj salary'.split()]
        pos_map = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 4, 'DST': 5}
        pos_num = [pos_map[pos] for pos in lineup['pos'].values]
        lineup['pos_num'] = pos_num
        lineup = lineup.sort_values('pos_num')
        lineup.drop('pos_num', axis=1, inplace=True)
        lineup = lineup.append(lineup.sum(numeric_only=True), ignore_index=True)
        return lineup, proj_pts

    def _save_lineups(self):
        path = f'../lineups/{self.year}/{self.week}/{self.league}/{self.budget}'
        mkdir(path)
        for i, lineup in self.lineups.items():
            lineup.to_csv(f'{path}/op_lineup_{i}.csv', index=False)

    def get_optimal_lineup(self, n_lineups=1, max_players_per_team=3,
                           save=False, verbose=False):
        self.max_players_per_team = max_players_per_team
        lineups = {}
        pt_lim=10000
        for i in range(n_lineups):
            lineup, proj = self._optimize(pt_lim=pt_lim)
            lineups[i+1] = lineup
            pt_lim = proj-0.1
            if verbose:
                print(f'Lineup #{i+1}')
                print(lineup.set_index('player'))
                print('----------------')

        self.lineups = lineups
        if save:
            self._save_lineups()

# %%


self = LineupOptimizer(year=2018, week=4, league='FanDuel')
self.get_optimal_lineup(n_lineups=3, max_players_per_team=3)
self.get_optimal_lineup(n_lineups=3, max_players_per_team=3, verbose=True)
