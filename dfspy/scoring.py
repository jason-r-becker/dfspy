import json
import numpy as np
import pandas as pd

# FanDuel_scoring = {
#     'Rush Yds': 0.1,
#     'Rush TD': 6.0,
#     'Pass Yds': 0.04,
#     'Pass TD': 4.0,
#     'Pass Int': -1.0,
#     'Rec Yds': 0.1,
#     'Rec TD': 6.0,
#     'Receptions': 0.5,
#     'Fum Lost': -2.0,
#     '2PT': 2.0,
#     'PAT Made': 1.0,
#     'FG 0-19': 3.0,
#     'FG 20-29': 3.0,
#     'FG 30-39': 3.0,
#     'FG 40-49': 4.0,
#     'Sack': 1.0,
#     'Fum Rec': 2.0,
#     'Return TD': 6.0,
#     'TD': 6.0,
#     'Saf': 2.0,
#     'Block': 2.0,
#     'Int': 2.0,
#     }
#
# DraftKings_scoring = {
#     'Rush Yds': 0.1,
#     'Rush TD': 6.0,
#     'Pass Yds': 0.04,
#     'Pass TD': 4.0,
#     'Pass Int': -1.0,
#     'Rec Yds': 0.1,
#     'Rec TD': 6.0,
#     'Receptions': 1.0,
#     'Fum Lost': -2.0,
#     '2PT': 2.0,
#     'PAT Made': 1.0,
#     'FG 0-19': 3.0,
#     'FG 20-29': 3.0,
#     'FG 30-39': 3.0,
#     'FG 40-49': 4.0,
#     'Sack': 1.0,
#     'Fum Rec': 2.0,
#     'Return TD': 6.0,
#     'TD': 6.0,
#     'Saf': 2.0,
#     'Block': 2.0,
#     'Int': 2.0,
#     }
#
# with open('../data/.FanDuel_scoring.json', 'w') as fid:
#     json.dump(FanDuel_scoring, fid, indent=4)
#
# with open('../data/.DraftKings_scoring.json', 'w') as fid:
#     json.dump(DraftKings_scoring, fid, indent=4)

def get_score(df, pos, league='FanDuel', type='proj'):
    """
    Calculate fantasy point projections for any position given league scoring
    rules and raw stat projections.
    """

    # load league scoring rules
    with open(f'../data/.{league}_scoring.json', 'r') as fid:
        stat_map = json.load(fid)

    # loop through possible stats to calculate fantasy point projection
    score = np.zeros(len(df))
    for stat, pts in stat_map.items():
        try:
            score += df[stat].values * stat_map[stat]
        except:
            pass

    if pos == 'DST':
        # interpolate fantasy points using distance from tier centers
        pts_allow = [0, 3.5, 10, 17, 24, 31, 38]
        ftsy_pts = [10, 7, 4, 1, 0, -1, -4]
        pts = df['Pts Allow'].values
        score += np.array([np.interp(pt, pts_allow, ftsy_pts) for pt in pts])


    # TODO: model probability of reaching milestone for DraftKings leagus

    df[type] = score
    final_df = df[['Player', 'Team', 'POS', type]].copy()
    final_df.columns = 'player team pos proj'.split()

    return final_df
#
# def scoring_DST(df):
#     sack, int, fum, saf, td, block, ret_td, pts = 1, 2, 2, 2, 6, 1, 6, -5
#     proj = (df['Sack'].values*sack +
#             df['Int'].values*int +
#             df['Fum Rec'].values*fum +
#             df['Saf'].values*saf +
#             df['TD'].values*td +
#             df['Block'].values*block +
#             df['Return TD'].values*ret_td +
#             df['Pts Allow'].values/pts)
#     df['proj'] = proj
#     df = df['Player Team POS proj'.split()]
#     df.columns = 'player team pos proj'.split()
#     return df
#
# def scoring_K(df):
#     pat, fg2, fg3, fg4 = 1, 3, 3, 4
#     proj = (df['PAT Made'].values*pat +
#             df['FG 0-19'].values*fg2 +
#             df['FG 20-29'].values*fg3 +
#             df['FG 30-39'].values*fg3 +
#             df['FG 40-49'].values*fg4)
#     df['proj'] = proj
#     df = df['Player Team POS proj'.split()]
#     df.columns = 'player team pos proj'.split()
#     return df
#
# def scoring_TE(df, ppr=0.5):
#     rec, rec_yds, rec_td = ppr, 10, 6
#     proj = (df['Receptions'].values*rec +
#             df['Rec Yds'].values/rec_yds +
#             df['Rec TD'].values*rec_td)
#     df['proj'] = proj
#     df = df['Player Team POS proj'.split()]
#     df.columns = 'player team pos proj'.split()
#     return df
#
# def scoring_WR(df, ppr=0.5):
#     rush_yds, rush_td, rec, rec_yds, rec_td, fum_lost = \
#         10, 6, ppr, 10, 6, -2
#     proj = (df['Rush Yds'].values/rush_yds +
#             df['Rush TD'].values*rush_td +
#             df['Receptions'].values*rec +
#             df['Rec Yds'].values/rec_yds +
#             df['Rec TD'].values*rec_td)
#     df['proj'] = proj
#     df = df['Player Team POS proj'.split()]
#     df.columns = 'player team pos proj'.split()
#     return df
#
# def scoring_RB(df, ppr=0.5):
#     rush_yds, rush_td, rec, rec_yds, rec_td, fum_lost, two_pt = \
#         10, 6, ppr, 10, 6, -2, 2
#     proj = (df['Rush Yds'].values/rush_yds +
#             df['Rush TD'].values*rush_td +
#             df['Receptions'].values*rec +
#             df['Rec Yds'].values/rec_yds +
#             df['Rec TD'].values*rec_td +
#             df['2PT'].values*two_pt)
#     df['proj'] = proj
#     df = df['Player Team POS proj'.split()]
#     df.columns = 'player team pos proj'.split()
#     return df
#
# def scoring_QB(df):
#     pass_yds, pass_td, pass_int, rush_yds, rush_td, fum_lost, two_pt = \
#         25, 4, -2, 10, 6, -2, 2
#     proj = (df['Pass Yds'].values/pass_yds +
#             df['Pass TD'].values*pass_td +
#             df['Pass Int'].values*pass_int +
#             df['Rush Yds'].values/rush_yds +
#             df['Rush TD'].values*rush_td +
#             df['Fum Lost'].values/fum_lost +
#             df['2PT'].values*two_pt)
#     df['proj'] = proj
#     df = df['Player Team POS proj'.split()]
#     df.columns = 'player team pos proj'.split()
#     return df
