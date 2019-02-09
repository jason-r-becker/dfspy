import numpy as np
import pandas as pd

def scoring_DST(df):
    sack, int, fum, saf, td, block, ret_td, pts = 1, 2, 2, 2, 6, 1, 6, -5
    proj = (df['Sack'].values*sack +
            df['Int'].values*int +
            df['Fum Rec'].values*fum +
            df['Saf'].values*saf +
            df['TD'].values*td +
            df['Block'].values*block +
            df['Return TD'].values*ret_td +
            df['Pts Allow'].values/pts)
    df['proj'] = proj.round(2)
    df = df['Player Team POS proj'.split()]
    df.columns = 'player team pos proj'.split()
    return df

def scoring_K(df):
    pat, fg2, fg3, fg4 = 1, 3, 3, 4
    proj = (df['PAT Made'].values*pat +
            df['FG 0-19'].values*fg2 +
            df['FG 20-29'].values*fg3 +
            df['FG 30-39'].values*fg3 +
            df['FG 40-49'].values*fg4)
    df['proj'] = proj.round(2)
    df = df['Player Team POS proj'.split()]
    df.columns = 'player team pos proj'.split()
    return df

def scoring_TE(df, ppr=0.5):
    rush_yds, rush_td, rec, rec_yds, rec_td = 10, 6, ppr, 10, 6
    proj = (df['Rush Yds'].values/rush_yds +
            df['Rush TD'].values/rush_td +
            df['Receptions'].values/rec +
            df['Rec Yds'].values/rec_yds +
            df['Rec TD'].values/rec_td)
    df['proj'] = proj.round(2)
    df = df['Player Team POS proj'.split()]
    df.columns = 'player team pos proj'.split()
    return df

def scoring_WR(df, ppr=0.5):
    rush_yds, rush_td, rec, rec_yds, rec_td, fum_lost = \
        10, 6, ppr, 10, 6, -2
    proj = (df['Rush Yds'].values/rush_yds +
            df['Rush TD'].values/rush_td +
            df['Receptions'].values/rec +
            df['Rec Yds'].values/rec_yds +
            df['Rec TD'].values/rec_td)
    df['proj'] = proj.round(2)
    df = df['Player Team POS proj'.split()]
    df.columns = 'player team pos proj'.split()
    return df

def scoring_RB(df, ppr=0.5):
    rush_yds, rush_td, rec, rec_yds, rec_td, fum_lost, two_pt = \
        10, 6, ppr, 10, 6, -2, 2
    proj = (df['Rush Yds'].values/rush_yds +
            df['Rush TD'].values/rush_td +
            df['Receptions'].values/rec +
            df['Rec Yds'].values/rec_yds +
            df['Rec TD'].values/rec_td +
            df['2PT'].values/two_pt)
    df['proj'] = proj.round(2)
    df = df['Player Team POS proj'.split()]
    df.columns = 'player team pos proj'.split()
    return df

def scoring_QB(df):
    pass_yds, pass_td, pass_int, rush_yds, rush_td, fum_lost, two_pt = \
        25, 4, -2, 10, 6, -2, 2
    proj = (df['Pass Yds'].values/pass_yds +
            df['Pass TD'].values/pass_td +
            df['Pass Int'].values/pass_int +
            df['Rush Yds'].values/rush_yds +
            df['Rush TD'].values/rush_td +
            df['Fum Lost'].values/fum_lost +
            df['2PT'].values/two_pt)
    df['proj'] = proj.round(2)
    df = df['Player Team POS proj'.split()]
    df.columns = 'player team pos proj'.split()
    return df
