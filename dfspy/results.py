import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scoring import get_score

plt.style.use('fivethirtyeight')

# %%
def get_season_data(
                    years=[2018],
                    weeks=list(range(1, 18)),
                    positions=['QB', 'RB', 'WR', 'TE', 'DST'],
                    league='FanDuel',
                    ):

    act = {}
    proj = {}
    for year in years:
        for week in weeks:
            for pos in positions:
                # projections
                fid = f'../data/{year}/{week}/{pos}/NFL.csv'
                projdf = pd.read_csv(fid)
                proj[(year, week, pos)] = get_score(
                                                    projdf,
                                                    pos=pos,
                                                    type='Proj',
                                                    league=league,
                                                    )
                # actual stats
                fid = f'../data/{year}/{week}/{pos}/STATS.csv'
                actdf = pd.read_csv(fid)
                act[(year, week, pos)] = get_score(
                                                   actdf, pos=pos,
                                                   type='Actual',
                                                   league=league,
                                                   )


    projdf = pd.concat(proj)
    projdf = projdf.reset_index().drop('level_2 level_3'.split(), axis=1)
    projdf.columns = 'year week player team pos proj'.split()
    projdf.set_index('year week player team pos'.split(), inplace=True)


    actdf = pd.concat(act)
    actdf = actdf.reset_index().drop('level_2 level_3'.split(), axis=1)
    actdf.columns = 'year week player team pos actual'.split()
    actdf.set_index('year week player team pos'.split(), inplace=True)

    df = projdf.join(actdf, how='left')
    df.dropna(inplace=True)
    df.reset_index(inplace=True)

    return df

# %%

def plot_comparison(
                    df,
                    t=3,
                    positions='QB RB WR TE DST'.split()
                    ):

    n = len(positions)
    fig, axes = plt.subplots(n, 1, figsize=(n, 5*n))
    for pos, ax  in zip(positions, axes.flat):
        scores = df['actual'][(
                              (df['pos']==pos)
                              & (df['proj']>t)
                              )].values
        projs = df['proj'][(
                           (df['pos']==pos)
                           & (df['proj']>t)
                           )].values
        ax.hist(scores, bins=np.arange(0, 50, 1), alpha=0.7,
                color='navy', label='Actual')
        ax.hist(projs, bins=np.arange(0, 50, 1), alpha=0.7,
                color='steelblue', label='Projections')
        ax.set_title(f'{pos} - {len(scores):,.0f} Samples', fontsize=12)
        ax.set_yticklabels([])
        ax.set_xlim(0, np.max(scores)+1)

    plt.legend()
    plt.suptitle(f'2018 - Actual vs Projected Fantasy Pts'
                 f'\n Projection > {t} Pts',
                 fontsize=16)
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.show()

# %%
