import argparse
import json
import os
from collections import defaultdict
from datetime import datetime as dt
from datetime import timedelta
from datetime import time
from tabulate import tabulate
from time import sleep, perf_counter

import cvxpy as cp
import empiricalutilities as eu
import numpy as np
import pandas as pd
from tqdm import tqdm

from scoring import get_score
from scrape_data import mkdir
from scrape_schedule import get_yearly_schedule
# %%

def main():
    args = parse_args()
    optimizer = LineupOptimizer(
        year=args.year,
        week=args.week,
        league=args.league,
        )

    optimizer.get_optimal_lineup(
        league=args.league,
        days=args.days,
        start=args.starttime,
        end=args.endtime,
        type=args.type,
        risk=args.risk,
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
    parser.add_argument('-d', '--days', help='Day(s) of games')
    parser.add_argument('-ts', '--starttime', help='Start of gametimes (start)')
    parser.add_argument('-te', '--endtime', help='End of gametimes (start)')
    parser.add_argument('-l', '--league', help='FanDuel, DraftKings, etc.')
    parser.add_argument('-n', '--n_lineups', type=int, help='Number of lineups')
    parser.add_argument('-m', '--max_players', type=int, help='Max plyrs/team')
    parser.add_argument('-st', '--stack', action='store_true', help='Stack?')
    parser.add_argument('-r', '--res', action='store_true', help='See result?')
    parser.add_argument('-s', '--save', action='store_true', help='Save?')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print?')
    parser.add_argument('-t', '--type', help="'actual' or 'proj'")
    parser.add_argument('-ri', '--risk', type=float, help="-1 to 1")

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
        days='Thu Sat Sun Mon',
        starttime='12:00AM',
        endtime='11:59PM',
        league='FanDuel',
        type='proj',
        risk=0,
        n_lineups=1,
        max_players=4,
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
        # self.schedule = get_yearly_schedule([year], [week])[year]
        self.schedule = pd.read_csv('week1schedule.csv', index_col=0)
        gametimes = self.schedule['gametime'].values
        times = [time(int(x.split(':')[0]), int(x.split(':')[1]))
                    for x in gametimes]
        self.schedule['gametime'] = times
        self.data, self.results = self._load_data()
        self.budg = {'FanDuel': 60000, 'DraftKings': 50000}[self.league]

    def __repr__(self):
        return f'Lineup Optimizer for week #{self.week}, {self.year}'

    def _read_file(self, pos):
        filepath = f'../data/{self.year}/{self.week}/{pos}/'
        srcs = 'FLOOR PROJ CEIL'.split()
        dfs = []
        for src in srcs:
            fid = filepath+src+'.csv'
            df = pd.read_csv(fid)
            df[src] = get_score(df, pos, self.league, type='proj')
            df = df[['Player', 'Team', src]].set_index(['Player', 'Team'])
            dfs.append(df)
        df = pd.concat(dfs, axis=1).reset_index()
        df['pos'] = pos
        df.columns = [col.lower() for col in df.columns]
        return df

    def _load_data(self):
        positions = 'QB RB WR TE DST'.split()
        player_dfs = {}
        results_dfs = {}
        for pos in positions:
            filepath = f'../data/{self.year}/{self.week}/{pos}/'
            player_dfs[pos] = self._read_file(pos)
            try:
                results_dfs[pos] = pd.read_csv(filepath+'STATS.csv')
            except:
                pass

        return player_dfs, results_dfs

    def _get_input_data(self, teams, type='proj'):
        positions = 'QB RB WR TE DST'.split()
        dfs = []
        for pos in positions:
            df = self.data[pos].copy()

            if type == 'actual':
                df1 = self.results[pos].copy()
                df1['actual'] = get_score(df1, pos, self.league, 'actual')
                df1.columns = [x.lower() for x in df1.columns]

                df = df.set_index('player team pos'.split()).join(
                        df1.set_index('player team pos'.split())
                        ).reset_index()

            filepath = f'../data/{self.year}/{self.week}/{pos}/'
            costs_df = pd.read_csv(filepath+self.league+'.csv')
            costs_df.columns = 'player team salary'.split()

            if pos == 'DST':
                jointdf = df.set_index('team').join(
                            costs_df.set_index('team')['salary'],
                            how='left',
                            sort=False,
                            rsuffix='_').dropna().reset_index()
                cols = list(jointdf)
                cols[0], cols[1] = 'player', 'team'
                jointdf = jointdf[cols].copy()
            else:
                jointdf = df.set_index(['player', 'team']).join(
                            costs_df.set_index(['player', 'team'])['salary'],
                            how='left', sort=False).dropna().reset_index()

            dfs.append(jointdf)

        fulldf = pd.concat(dfs, axis=0, sort=False).reset_index(drop=True)
        fulldf1 = fulldf.loc[fulldf['team'].isin(teams)].copy()
        pos_dummies = pd.get_dummies(fulldf1['pos'])
        team_dummies = pd.get_dummies(fulldf1['team'])
        finaldf = fulldf1.join([pos_dummies, team_dummies],
                    how='left').sort_values(['pos', 'team']).copy()

        return finaldf.reset_index(drop=True)

    def _get_results(self, lineup):
        positions = 'QB RB WR TE DST'.split()
        res = []
        for pos in positions:
            df = self.results[pos].copy()
            df['actual'] = get_score(self.results[pos], pos,
                                     self.league, 'actual')
            df.columns = [col.lower() for col in df.columns]
            df = df['player team pos actual'.split()]
            res.append(df)

        results = pd.concat(res).set_index('player team pos'.split())
        lp = lineup.set_index('player team pos'.split()).copy()
        lp = lp.join(results).reset_index()
        lp['actual'] = lp['actual'].replace(np.nan, lp['actual'].sum())

        return lp

    def _valid_teams(self, days=['Thu', 'Sat', 'Sun', 'Mon'],
                     start='12:00AM', end='11:59PM'):
        """
        Returns list of teams playing on the specified day, within the specified
        time period.

        day: str: One of 'Thu', 'Sat', 'Sun', 'Mon'
        start: str: '1:00PM', '9:30AM', '4:25PM', ...
        end: str: same as start
        """

        df = self.schedule
        df1 = df.loc[df['game_day_of_week'].isin(days)].copy()
        # start time
        hour = int(start.split(':')[0])
        min = int(start.split(':')[1][:2])
        M = start[-2:]
        tsStart = time(hour+12, min) if M == 'PM' else time(hour, min)
        # end time
        hour = int(end.split(':')[0])
        min = int(end.split(':')[1][:2])
        M = end[-2:]
        tsEnd = time(hour+12, min) if M == 'PM' else time(hour, min)

        df2 = df1[(df1['gametime']>=tsStart)&(df1['gametime']<tsEnd)].copy()
        teams = list(df2['team'].unique())
        with open(f'../data/.team_mappings.json', 'r') as fid:
            team_map = json.load(fid)
        teams = [team_map[team] for team in teams]
        return teams


    def _save_lineups(self, lineup, risk):
        path = f'../lineups/{self.year}/{self.week}/{self.league}/{self.budg}'
        mkdir(path)
        lineup.to_csv(f'{path}/op_{risk}_{i}.csv', index=False)

    def _format_lineup(self, data, idxs):
        cols = 'player team pos salary floor risk ceil'.split()

        proj_pts = data.iloc[idxs]['floor risk ceil'.split()].sum()
        lineup = data.iloc[idxs][cols].copy()
        pos_map = {'QB': 5, 'RB': 4, 'WR': 3, 'TE': 2, 'DST': 1}
        pos_num = [pos_map[pos] for pos in lineup['pos'].values]
        lineup['pos_num'] = pos_num
        lineup = lineup.sort_values(['pos_num', 'risk'], ascending=False)
        lineup.drop('pos_num', axis=1, inplace=True)
        lineup = lineup.append(lineup.sum(numeric_only=True),
                               ignore_index=True)
        lineup['player'] = lineup['player'].replace(np.nan, 'Total')

        return lineup

    def _optimize(self, data, mppt, stack, n_lineups, type, risk):

        if risk == 0:
            data['risk'] = data['proj']
        elif (risk > 0) & (risk <= 1):
            data['risk'] = data['ceil']*risk + (1-risk)*data['proj']
        elif (risk < 0) & (risk >= -1):
            data['risk'] = data['floor']*abs(risk) + (1-abs(risk))*data['proj']
        else:
            raise ValueError('risk must be between -1 and 1')

        full_data = data.copy()
        N = len(full_data)
        np.random.seed(0)
        lineup_set = []
        lineups = []
        while len(lineup_set) != n_lineups:
            if n_lineups > 1:
                idxs_to_drop = np.random.choice(int(N), int(N*0.25))
                data = full_data.drop(full_data.index[idxs_to_drop],
                    axis=0).reset_index().copy()
            QBs = data[data['pos']=='DST'].index.max()+1
            teams = data['team'].unique()
            n = len(data)
            W = cp.Variable((n, 1), boolean=True)

            constraints = [
                cp.matmul(W.T,
                    data['salary'].values.reshape(-1, 1)) <= self.budg,
                cp.sum(W)==9,
                cp.matmul(W.T, data['QB'].values.reshape(-1, 1)) == 1,
                cp.matmul(W.T, data['RB'].values.reshape(-1, 1)) <= 3,
                cp.matmul(W.T, data['RB'].values.reshape(-1, 1)) >= 2,
                cp.matmul(W.T, data['WR'].values.reshape(-1, 1)) <= 4,
                cp.matmul(W.T, data['WR'].values.reshape(-1, 1)) >= 3,
                cp.matmul(W.T, data['TE'].values.reshape(-1, 1)) <= 2,
                cp.matmul(W.T, data['TE'].values.reshape(-1, 1)) >= 1,
                cp.matmul(W.T, data['DST'].values.reshape(-1, 1)) == 1,
                cp.max(cp.matmul(W[QBs:].T,
                    data[teams].values[QBs:, :])) <= mppt,
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
            if type == 'actual':
                obj = cp.Maximize(cp.matmul(W.T,
                        data['actual'].values.reshape(-1, 1)))
            else:
                obj = cp.Maximize(cp.matmul(W.T,
                        data['risk'].values.reshape(-1, 1)))
            prob = cp.Problem(obj, constraints)
            prob.solve(solver='GLPK_MI')
            idx = []
            W.value = W.value.round()
            for i, w in enumerate(W.value):
                if w == 1:
                    idx.append(i)
            lp = self._format_lineup(data, idx)
            players = tuple(lp['player'].values)
            if players not in set(lineup_set):
                lineups.append(lp)
                lineup_set.append(players)
            else:
                continue
            sleep(0.01)
        return lineups

    def get_optimal_lineup(
        self,
        type='proj',
        risk=0,
        n_lineups=1,
        mppt=4,
        stack=False,
        result=False,
        save=False,
        verbose=False,
        days='Thu Sat Sun Mon',
        start='12:00AM',
        end='11:59PM',
        ):

        teams = self._valid_teams(days.split(), start, end)
        data = self._get_input_data(teams, type).copy()
        filt = {
            'DST': -100.0,
            'QB': 5.0,
            'RB': 4.0,
            'WR': 4.0,
            'TE': 3.0,
            }
        temps = []
        for pos, pts in filt.items():
            temp = data[data['pos'] == pos].copy()
            temp = temp[temp['proj'] >= pts]
            temps.append(temp)
        data = pd.concat(temps).sort_values(['pos', 'proj', 'salary'])

        lineups = self._optimize(
            data=data,
            type=type,
            risk=risk,
            n_lineups=n_lineups,
            mppt=mppt,
            stack=stack,
            )

        self.lineups = []
        for i, lineup in enumerate(lineups):
            lp = self._get_results(lineup) if result else lineup.copy()

            if verbose:
                print('------------------------')
                print(f'Lineup #{i+1}, Week {self.week}, {self.year}')
                print(
                        tabulate(
                            lp.set_index('player'),
                            headers='keys',
                            tablefmt='psql',
                            floatfmt=['', '', '', ',.0f', '.2f', '.2f',
                                        '.2f', '.2f']
                            )
                        )
            self.lineups.append(lp)
            if save:
                self._save_lineups(lp, risk)

# %%
# self = LineupOptimizer(2018, 1, 'DraftKings')
# self.get_optimal_lineup(verbose=True, risk=-0.5, result=True)

# %%
if __name__ == '__main__':
    main()
