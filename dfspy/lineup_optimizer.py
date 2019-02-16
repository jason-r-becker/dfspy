import argparse
import os
from datetime import datetime as dt
from datetime import timedelta

import cvxpy as cp
import empiricalutilities as eu
import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

from scoring import get_score
from scrape_data import mkdir
from results import get_season_data
# %%

def main():
    args = parse_args()
    optimizer = LineupOptimizer(
        year=args.year,
        week=args.week,
        league=args.league,
        )
    optimizer.get_optimal_lineup(
        n_lineups=args.n_lineups,
        max_players_per_team=args.max_players,
        stack=args.stack,
        result=args.res,
        save=args.save,
        verbose=args.verbose,
        )


def parse_args():
    """Collect settings from command line and set defaults"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--year', type=int, help='Year of season')
    parser.add_argument('-w', '--week', type=int, help='Week of season')
    parser.add_argument('-l', '--league', help='FanDuel, DraftKings, etc.')
    parser.add_argument('-n', '--n_lineups', type=int, help='Number of lineups')
    parser.add_argument('-m', '--max_players', type=int, help='Max plyrs/team')
    parser.add_argument('-st', '--stack', action='store_true', help='Stack?')
    parser.add_argument('-r', '--res', action='store_true', help='See result?')
    parser.add_argument('-s', '--save', action='store_true', help='Save?')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print?')

    default_league = 'FanDuel'
    today = dt.utcnow()
    default_year = today.year if today.month > 7 else today.year - 1
    season_start = {  # Date of Thursday night game starting each season.
        2015: '9/10/2015',
        2016: '9/8/2016',
        2017: '9/7/2017',
        2018: '9/6/2018',
        2019: '9/5/2019',
        }[default_year]

    # Default week is the current NFL season week, starting on Tuesdays.
    # Before the season default is week 1, and after the season it is 17.
    default_week = int(np.ceil(
        (today-pd.to_datetime(season_start)+timedelta(days=2)).total_seconds()
        / (3600 * 24 * 7)
        ))
    default_week = max(1, min(17, default_week))

    # set default arguments
    parser.set_defaults(
        year=default_year,
        week=default_week,
        league=default_league,
        n_lineups=1,
        max_players=3,
        stack=False,
        res=False,
        save=False,
        verbose=False,
        )
    args = parser.parse_args()

    return args

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
        self.resdf = self._load_results()
        self._pt_lim = 10000
        self.max_players_per_team = 3

    def _load_data(self, source='NFL'):
        POS = 'QB RB WR TE DST'.split()

        player_dfs = {}
        for pos in POS:
            filepath = f'../data/{self.year}/{self.week}/{pos}/'
            df = pd.read_csv(filepath+source+'.csv')
            df = get_score(df, pos=pos, league=self.league)
            df.columns = 'player team pos proj'.split()

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

    def _optimize(self):
        N = len(self.data)
        W = cp.Variable((N, 1), boolean=True)

        DSTe = self.data[self.data['pos']=='DST'].index.max()+1
        WRs = self.data[self.data['pos']=='WR'].index.min()
        WRe = self.data[self.data['pos']=='WR'].index.max()+1
        QBs = self.data[self.data['pos']=='QB'].index.min()
        QBe = self.data[self.data['pos']=='QB'].index.max()+1

        teams = self.data['team'].unique()
        constrs = [
            cp.matmul(W.T, self._data_col_vector('salary')) <= self.budget,
            cp.matmul(W.T, self._data_col_vector('proj')) <= self._pt_lim,
            cp.sum(W)==9,
            cp.matmul(W.T, self._data_col_vector('QB')) == 1,
            cp.matmul(W.T, self._data_col_vector('RB')) <= 3,
            cp.matmul(W.T, self._data_col_vector('RB')) >= 2,
            cp.matmul(W.T, self._data_col_vector('WR')) <= 4,
            cp.matmul(W.T, self._data_col_vector('WR')) >= 3,
            cp.matmul(W.T, self._data_col_vector('TE')) <= 2,
            cp.matmul(W.T, self._data_col_vector('TE')) >= 1,
            cp.matmul(W.T, self._data_col_vector('DST')) == 1,
            cp.max(cp.matmul(W[DSTe:].T,
                self.data[teams].values[DSTe:, :])) <= self.max_players_per_team,
            ]

        if self._stack:
            constrs.append(cp.norm(
                    cp.matmul(W[WRs:WRe].T, self.data[teams].values[WRs:WRe]) -
                    cp.matmul(W[QBs:QBe].T, self.data[teams].values[QBs:QBe]),
                    2) <= np.sqrt(3)
                    )

        obj = cp.Maximize(cp.matmul(W.T, self._data_col_vector('proj')))
        prob = cp.Problem(obj, constrs)
        prob.solve()
        W.value = W.value.round()
        idx = []
        for i, w in enumerate(W.value):
            if w == 1:
                idx.append(i)
        proj_pts = self.data.iloc[idx]['proj'].sum()
        lineup = self.data.iloc[idx]['player team pos salary proj'.split()]
        pos_map = {'QB': 5, 'RB': 4, 'WR': 3, 'TE': 2, 'DST': 1}
        pos_num = [pos_map[pos] for pos in lineup['pos'].values]
        lineup['pos_num'] = pos_num
        lineup = lineup.sort_values('pos_num proj'.split(), ascending=False)
        lineup.drop('pos_num', axis=1, inplace=True)
        lineup = lineup.append(lineup.sum(numeric_only=True), ignore_index=True)
        lineup['player'] = lineup['player'].replace(np.nan, 'Total')
        return lineup, proj_pts

    def _save_lineups(self):
        path = f'../lineups/{self.year}/{self.week}/{self.league}/{self.budget}'
        mkdir(path)
        for i, lineup in self.lineups.items():
            lineup.to_csv(f'{path}/op_lineup_{i}.csv', index=False)

    def get_optimal_lineup(self, n_lineups=1, max_players_per_team=3,
                           stack=False, result=False,
                           save=False, verbose=False):

        self.max_players_per_team = max_players_per_team
        self._stack = stack
        lineups = {}
        for i in range(n_lineups):
            lineup, proj = self._optimize()
            lineups[i+1] = self.get_results(lineup) if result else lineup
            self._pt_lim = proj-0.1
            if verbose:
                print('------------------------')
                print(f'Lineup #{i+1}, Week {self.week}, {self.year}')
                print(
                        tabulate(
                            lineups[i+1].set_index('player'),
                            headers='keys',
                            tablefmt='psql',
                            floatfmt=['', '', '', ',.0f', '.2f', '.2f']
                            )
                        )

        self.lineups = lineups
        if save:
            self._save_lineups()

    def _load_results(self):
        try:
            resdf = get_season_data(
                                    years=[self.year],
                                    weeks=[self.week],
                                    league=self.league,
                                    )
        except:
            raise ValueError('No past data for this week')

        return resdf

    def get_results(self, lineup):

        resdf = self.resdf.set_index('player team pos'.split())

        # get results for lineup
        result_df = lineup.set_index('player team pos'.split()).join(
                            resdf['actual'], how='left'
                            ).dropna()
        result_df = result_df.reset_index()
        result_df = result_df.append(result_df.sum(numeric_only=True),
                                     ignore_index=True)
        result_df['player'] = result_df['player'].replace(np.nan, 'Total')
        return result_df

# %%


# self = LineupOptimizer(year=2018, week=16, league='FanDuel')
# self.get_optimal_lineup(stack=True, verbose=True, result=True)

if __name__ == '__main__':
    main()
