import numpy as np
import cvxpy as cp
import pandas as pd
from scoring import *
# %%

def main():
    year = int(input('Enter Year: '))
    week = int(input('Enter Week: '))
    budget = int(input('Enter Budget: '))
    source = 'NFL'
    print(f'Source = {source}')
    df = read_data(year=year, week=week, source=source)
    df = get_costs(df)
    lineup, proj_pts, cost = get_optimal_lineup(df, budget)
    print('---------- \n Lineup: \n', lineup)
    print('---------- \n Projected Points: \n', proj_pts)
    print(f'--------- \n Cost={cost}, Budget={budget}, Cap Room={budget-cost}')
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
    return df
# make random costs
def get_costs(df):
    N = len(df)
    costs = np.random.randint(low=10, high=600, size=N)*10
    df['cost'] = costs
    df.set_index('player', inplace=True)
    return df

def get_optimal_lineup(df, budget):
    N = len(df)
    W = cp.Variable((N, 1), boolean=True)
    constrs = [cp.sum(cp.multiply(W, df['cost'].values.reshape(N, 1)))<=budget,
                   cp.sum(W)<=9,
                   cp.sum(cp.multiply(W, df['QB'].values.reshape(N, 1)))==1,
                   cp.sum(cp.multiply(W, df['RB'].values.reshape(N, 1)))<=3,
                   cp.sum(cp.multiply(W, df['WR'].values.reshape(N, 1)))<=3,
                   cp.sum(cp.multiply(W, df['TE'].values.reshape(N, 1)))<=2,
                   cp.sum(cp.multiply(W, df['TE'].values.reshape(N, 1)))>=1,
                   cp.sum(cp.multiply(W, df['K'].values.reshape(N, 1)))==1,
                   cp.sum(cp.multiply(W, df['DST'].values.reshape(N, 1)))==1]

    obj = cp.Maximize(cp.sum(cp.multiply(W, df['proj'].values.reshape(N, 1))))
    prob = cp.Problem(obj, constrs)
    prob.solve()
    W.value = W.value.round()
    idx = []
    for i, w in enumerate(W.value):
        if w == 1:
            idx.append(i)
    proj_pts = round(df.iloc[idx]['proj'].sum(),2)
    lineup_cost = df.iloc[idx]['cost'].sum()
    lineup = df.iloc[idx]['team pos proj cost'.split()]
    pos_map = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 4, 'K': 5, 'DST': 6}
    pos_num = [pos_map[pos] for pos in lineup['pos'].values]
    lineup['pos_num'] = pos_num
    lineup = lineup.sort_values('pos_num')
    lineup.drop('pos_num', axis=1, inplace=True)
    return lineup, proj_pts, lineup_cost

# %%
if __name__ == '__main__':
    main()
