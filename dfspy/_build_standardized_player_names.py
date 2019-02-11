"""
Create standardized player name .json file from NFL.com scrapes.
"""

import json
from glob import glob
from itertools import chain

import pandas as pd


def main():
    positions = 'QB RB WR TE K DST'.split()
    
    names = {}
    for pos in positions:
        # Find all NFL.com files for position.
        fids = glob(f'../data/**/{pos}/NFL.csv', recursive=True)
        fids.extend(glob(f'../data/**/{pos}/STATS.csv', recursive=True))
        
        # Read files and build list of unique player names for positon.
        names[pos] = sorted(list(set(chain.from_iterable([
            pd.read_csv(fid).Player.values for fid in fids]))))

   
    # Output to json file.
    with open('../data/.player_names.json', 'w') as fid:
        json.dump(names, fid, indent=4)


if __name__ == '__main__':
    main()
