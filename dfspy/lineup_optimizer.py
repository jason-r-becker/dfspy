import argparse
import os
from collections import defaultdict
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
        )

    optimizer.get_optimal_lineup(
        league=args.league,
        n_lineups=args.n_lineups,
        mppt=args.max_players,
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
        league='FanDuel',
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
    def __init__(self, year, week):
        self.year = year
        self.week = week
        self.data = self._load_data()

    def __repr__(self):
        return f'Lineup Optimizer for week #{self.week}, {self.year}'

    def _load_data(self, source='NFL'):
        positions = 'QB RB WR TE DST'.split()
        player_dfs = {}
        for pos in positions:
            filepath = f'../data/{self.year}/{self.week}/{pos}/'
            df = pd.read_csv(filepath+source+'.csv')
            player_dfs[pos] = df

        return player_dfs

    def _get_projections(self, league):
        positions = 'QB RB WR TE DST'.split()
        dfs = []
        for pos in positions:
            df = self.data[pos].copy()
            df = get_score(df, pos=pos, league=league).copy()

            filepath = f'../data/{self.year}/{self.week}/{pos}/'
            costs_df = pd.read_csv(filepath+league+'.csv')
            costs_df.columns = 'player team salary'.split()

            if pos == 'DST':
                jointdf = df.set_index('team').join(
                            costs_df.set_index('team')['salary'],
                            how='left',
                            sort=False,
                            rsuffix='_').dropna().reset_index()
                cols = 'player team pos proj salary'.split()
                jointdf = jointdf[cols].copy()
            else:
                jointdf = df.set_index(['player', 'team']).join(
                            costs_df.set_index(['player', 'team'])['salary'],
                            how='left', sort=False).dropna().reset_index()

            dfs.append(jointdf)

        fulldf = pd.concat(dfs, axis=0).reset_index(drop=True)
        pos_dummies = pd.get_dummies(fulldf['pos'])
        team_dummies = pd.get_dummies(fulldf['team'])
        finaldf = fulldf.join([pos_dummies, team_dummies],
                    how='left').sort_values(['pos', 'team']).copy()

        return finaldf.reset_index(drop=True)

    def _load_results(self, league):
        try:
            results = get_season_data(
                                    years=[self.year],
                                    weeks=[self.week],
                                    league=league,
                                    )
        except:
            raise ValueError('No past data for this week')

        cols = 'player team pos actual'.split()
        return results[cols].set_index('player team pos'.split()).copy()

    def _get_results(self, lineup, league):
        results = self._load_results(league)
        # get results for lineup
        result_df = lineup.set_index('player team pos'.split()).join(
                        results, how='left').dropna().reset_index()
        result_df = result_df.append(result_df.sum(numeric_only=True),
                                     ignore_index=True)
        result_df['player'] = result_df['player'].replace(np.nan, 'Total')
        return result_df

    def _save_lineups(self, lineup):
        path = f'../lineups/{self.year}/{self.week}/{self.league}/{self.budget}'
        mkdir(path)
        lineup.to_csv(f'{path}/op_lineup_{i}.csv', index=False)


    def _format_lineup(self, data, lineups):
        cols = 'player team pos salary proj'.split()
        formatted_lineups = []
        for idxs in lineups:
            proj_pts = data.iloc[idxs]['proj'].sum()
            lineup = data.iloc[idxs][cols].copy()
            pos_map = {'QB': 5, 'RB': 4, 'WR': 3, 'TE': 2, 'DST': 1}
            pos_num = [pos_map[pos] for pos in lineup['pos'].values]
            lineup['pos_num'] = pos_num
            lineup = lineup.sort_values('pos_num proj'.split(), ascending=False)
            lineup.drop('pos_num', axis=1, inplace=True)
            lineup = lineup.append(lineup.sum(numeric_only=True), ignore_index=True)
            lineup['player'] = lineup['player'].replace(np.nan, 'Total')
            formatted_lineups.append(lineup)
        return formatted_lineups

    def _problem_setup(self, data, league, mppt, stack, proj='proj'):
        # QB index start for stacking rule
        QBs = data[data['pos']=='DST'].index.max()+1
        teams = data['team'].unique()

        budget = {
            'FanDuel': 60000,
            'DraftKings': 50000,
            'ESPN': None,
            }[league]

        N = len(data)
        W = cp.Variable((N, 1), boolean=True)

        constraints = [
            cp.matmul(W.T, data['salary'].values.reshape(-1, 1)) <= budget,
            cp.sum(W)==9,
            cp.matmul(W.T, data['QB'].values.reshape(-1, 1)) == 1,
            cp.matmul(W.T, data['RB'].values.reshape(-1, 1)) <= 3,
            cp.matmul(W.T, data['RB'].values.reshape(-1, 1)) >= 2,
            cp.matmul(W.T, data['WR'].values.reshape(-1, 1)) <= 4,
            cp.matmul(W.T, data['WR'].values.reshape(-1, 1)) >= 3,
            cp.matmul(W.T, data['TE'].values.reshape(-1, 1)) <= 2,
            cp.matmul(W.T, data['TE'].values.reshape(-1, 1)) >= 1,
            cp.matmul(W.T, data['DST'].values.reshape(-1, 1)) == 1,
            cp.max(cp.matmul(W[QBs:].T, data[teams].values[QBs:, :])) <= mppt,
            ]

        if stack:
            WRs = data[data['pos']=='WR'].index.min()
            WRe = data[data['pos']=='WR'].index.max()+1
            QBe = data[data['pos']=='QB'].index.max()+1

            constraints.append(cp.norm(
                    cp.matmul(W[WRs:WRe].T, data[teams].values[WRs:WRe]) -
                    cp.matmul(W[QBs:QBe].T, data[teams].values[QBs:QBe]),
                    2) <= np.sqrt(3)
                    )

        obj = cp.Maximize(
            cp.matmul(W.T, data[proj].values.reshape(-1, 1)))

        return W, obj, constraints

    def _optimize(self, data, league, mppt, stack, n_lineups):
        W, obj, constraints = self._problem_setup(
                                data,
                                league,
                                mppt,
                                stack,
                                'proj'
                                )

        constraints.append(
            cp.matmul(W.T, data['proj'].values.reshape(-1, 1)) <= 10000
            )

        lineups = []
        for i in range(n_lineups):
            prob = cp.Problem(obj, constraints)
            prob.solve()

            idx = []
            W.value = W.value.round()
            for i, w in enumerate(W.value):
                if w == 1:
                    idx.append(i)

            lineups.append(idx)
            # update pt limit to make new unique lineup
            point_limit = prob.value - 0.1
            constraints[-1] = cp.matmul(W.T,
                data['proj'].values.reshape(-1, 1)) <= point_limit

        return lineups

    def get_optimal_lineup(
        self,
        league='FanDuel',
        n_lineups=1,
        mppt=3,
        stack=False,
        result=False,
        save=False,
        verbose=False,
        ):

        data = self._get_projections(league).copy()
        lineups = self._optimize(data, league, mppt, stack, n_lineups)
        fmt_lineups = self._format_lineup(data, lineups)

        for i, lineup in enumerate(fmt_lineups):
            lp = self._get_results(lineup, league) if result else lineup

            if verbose:
                print('------------------------')
                print(f'Lineup #{i+1}, Week {self.week}, {self.year}')
                print(
                        tabulate(
                            lp.set_index('player'),
                            headers='keys',
                            tablefmt='psql',
                            floatfmt=['', '', '', ',.0f', '.2f', '.2f']
                            )
                        )
            if save:
                self._save_lineup(lp)


# self = LineupOptimizer(2018, 4)
# self.get_optimal_lineup(verbose=True, stack=True, result=True)
# self.get_optimal_lineup(verbose=True, stack=False, n_lineups=3, mppt=1)

# %%
if __name__ == '__main__':
    main()
