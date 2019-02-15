import os
import json
import numpy as np
import pandas as pd
# %%

def main():
    years = [2016, 2017, 2018]
    weeks = list(range(1, 18))
    positions = 'QB RB WR TE K'.split()

    team_names = []
    for year in years:
        for week in weeks:
            for pos in positions:
                filepath = f'./{year}/{week}/{pos}/'
                files = os.listdir(filepath)
                for file in files:
                    filename = filepath+file
                    src = file.split('.')[0]
                    if src not in set(['Yahoo', 'NFL', 'FantasyPros',
                                       'ESPN', 'CBS', 'FFToday']):
                        continue
                    team_names = d[src](filename, team_names)

    team_names = set(team_names)
    team_names.remove(np.nan)
    tn = {k: 'xxx' for k in team_names}

    with open('./._team_mapping.json', 'w') as fid:
        json.dump(tn, fid, indent=4, sort_keys=True)

    return
# %%
def read_Yahoo(filename, team_names):
    df = pd.read_csv(filename, usecols=[0])
    for row in df.values:
        name = row[0].split('-')[0].split()[-1]
        team_names.append(name)
    return team_names

def read_NFL(filename, team_names):
    df = pd.read_csv(filename, usecols=[1])
    for name in df.values:
        team_names.append(name[0])
    return team_names

def read_FantasyPros(filename, team_names):
    df = pd.read_csv(filename, usecols=[0])
    for name in df.values:
        team_names.append(name[0].split()[-1])
    return team_names

def read_ESPN(filename, team_names):
    df = pd.read_csv(filename, usecols=[0])
    for name in df.values:
        team_names.append(name[0].split(',')[1].split()[0])
    return team_names

def read_CBS(filename, team_names):
    df = pd.read_csv(filename, usecols=[0])
    for name in df.values:
        team_names.append(name[0].split(',')[1].split()[0])
    return team_names

def read_FFToday(filename, team_names):
    df = pd.read_csv(filename, usecols=[1])
    for name in df.values:
        team_names.append(name[0])
    return team_names

d = {
    'Yahoo': read_Yahoo,
    'NFL': read_NFL,
    'FantasyPros': read_FantasyPros,
    'ESPN': read_ESPN,
    'CBS': read_CBS,
    'FFToday': read_FFToday
    }

# %%
if __name__ == '__main__':
    main()


# with open('../data/.team_mappings.json', 'r') as fid:
#     team_map = json.load(fid)
#
# new_keys = []
# vals = []
# for k, v in team_map.items():
#     if len(k)>3:
#         new_keys.append(k.rsplit()[-1])
#         new_keys.append(' '.join(k.rsplit()[:-1]))
#         vals.append(v)
#         vals.append(v)
#
# new_dict = {k: v for k, v in zip(new_keys, vals)}
# final_dict = {**team_map, **new_dict}
#
# with open('../data/.team_mappings.json', 'w') as fid:
#     json.dump(final_dict, fid, indent=4, sort_keys=True)
#
#
#
# for k, v in team_map.copy().items():
#     lsplit, rsplit = tuple(k.rsplit(' ', 1)) if len(k) > 3 else k, k
#     team_map[lsplit[0]] = v
#     team_map[rsplit[0]] = v
#
# final_dict = {}
# for k, v in sorted(team_map.items(), key=lambda x: x[1]):
#     final_dict[k] = v
