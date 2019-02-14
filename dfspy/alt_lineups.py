import os
import numpy as np
import cvxpy as cp
import pandas as pd
from scoring import *
# %%

def get_diverse_teams_lineup(df, budget, pt_lim, teams):
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
               cp.matmul(W.T, df['DST'].values.reshape(N, 1))==1,
               cp.max(cp.matmul(W.T, df.iloc[:, 10:-1]))<=1]

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

def get_cust_team_stack(df, budget, pt_lim, teams, nums):
    """
    allow for specification of which teams to stack
    Parameters:
    teams: list(str) ['NE', 'GB']
    nums: list(int) [2, 2]

    Example call:
    get_cust_team_stack(df, 10000, 1000, ['NE', 'GB', 'NO'], [3, 2, 2])
    """
    if np.sum(nums)>9:
        raise ValueError('Too many players specified')

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
    for t, n in zip(teams, nums):
        constrs.append(cp.matmul(W.T, df[t].values.reshape(N, 1))==n)

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
