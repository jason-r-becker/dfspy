import os
import numpy as np
import cvxpy as cp
import pandas as pd
from scoring import *
# %%

def main():
    year = int(input('Enter Year: '))
    week = int(input('Enter Week: '))
    budget = int(input('Enter Budget: '))
    n = int(input('Enter # of Lineups: '))
    save = input('Save? [y/n]: ')
    source = 'NFL'
    print(f'{year}, {week}')
    df = read_data(year=year, week=week, source=source)
    df = get_costs(df)

    # ensure destination exists
    file_dest = f'../lineups/{year}'
    if not os.path.exists(file_dest):
        os.mkdir(file_dest)

    file_dest = f'../lineups/{year}/{week}'
    if not os.path.exists(file_dest):
        os.mkdir(file_dest)

    file_dest = f'../lineups/{year}/{week}/{source}'
    if not os.path.exists(file_dest):
        os.mkdir(file_dest)

    # construct top n lineups and save to csv
    pt_lim = 1000
    for i in range(n):
        lineup, proj = get_optimal_lineup(df, budget, pt_lim)
        # consider adding budget to the filename or another folder layer
        if save == 'y':
            lineup.to_csv(f'{file_dest}/op_lineup_{i+1}.csv', index=False)
        pt_lim = proj-0.1

    return

def read_data(year, week, source):
    POS = 'QB RB WR TE K DST'.split()
    d = {'QB': scoring_QB,
         'RB': scoring_RB,
         'WR': scoring_WR,
         'TE': scoring_TE,
         'K': scoring_K,
         'DST': scoring_DST}
    player_dfs = {}
    for pos in POS:
        filepath = f'../data/{year}/{week}/{pos}/'
        df = pd.read_csv(filepath+source+'.csv')
        df = d[pos](df)
        player_dfs[pos] = df
    df = pd.concat(player_dfs).reset_index(drop=True)
    df = df.join(pd.get_dummies(df['pos']), how='left')
    df = df.join(pd.get_dummies(df['team']), how='left')

    return df
# make random costs
def get_costs(df):
    N = len(df)
    costs = np.random.randint(low=10, high=600, size=N)*10
    df['cost'] = costs
    #df.set_index('player', inplace=True)

    return df

def get_optimal_lineup(df, budget, pt_lim):
    N = len(df)
    W = cp.Variable((N, 1), boolean=True)
    constrs = [cp.matmul(W.T, df['cost'].values.reshape(N, 1))<=budget,
               cp.matmul(W.T, df['proj'].values.reshape(N, 1))<=pt_lim,
               cp.sum(W)==9,
               cp.matmul(W.T, df['QB'].values.reshape(N, 1))==1,
               cp.matmul(W.T, df['RB'].values.reshape(N, 1))<=3,
               cp.matmul(W.T, df['WR'].values.reshape(N, 1))<=3,
               cp.matmul(W.T, df['TE'].values.reshape(N, 1))<=2,
               cp.matmul(W.T, df['TE'].values.reshape(N, 1))>=1,
               cp.matmul(W.T, df['K'].values.reshape(N, 1))==1,
               cp.matmul(W.T, df['DST'].values.reshape(N, 1))==1]

    obj = cp.Maximize(cp.matmul(W.T, df['proj'].values.reshape(N, 1)))
    prob = cp.Problem(obj, constrs)
    prob.solve()
    W.value = W.value.round()
    idx = []
    for i, w in enumerate(W.value):
        if w == 1:
            idx.append(i)
    proj_pts = df.iloc[idx]['proj'].sum()
    lineup = df.iloc[idx]['player team pos proj cost'.split()]
    pos_map = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 4, 'K': 5, 'DST': 6}
    pos_num = [pos_map[pos] for pos in lineup['pos'].values]
    lineup['pos_num'] = pos_num
    lineup = lineup.sort_values('pos_num')
    lineup.drop('pos_num', axis=1, inplace=True)
    lineup = lineup.append(lineup.sum(numeric_only=True), ignore_index=True)

    return lineup, proj_pts

# %%
if __name__ == '__main__':
    main()
