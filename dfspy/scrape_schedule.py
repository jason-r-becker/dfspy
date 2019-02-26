from collections import defaultdict
from datetime import time

import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment

# %%

def get_schedule(years=[2018], weeks=[17]):
    """
    Function to return a dictionary of a dictionary of dictionaries with outer
    key as year number, inner key as week number and innermost key as date of
    game. Each innermost dictionary's value is a list of teams playing on that
    exact date.

    For example, structure of get_schedule()[2018]:
    {1: {
        '2018-09-06': ['ATL', 'PHI'],
        '2018-09-09': ['BUF', 'BAL', 'JAX', 'NYG', 'TB', ...],
        '2018-09-10': ['NYJ', 'DET', 'LA', 'OAK'],
        }}

    TODO: Add actual game time or primetime flag using the below class_ string:
        'schedules-list-matchup post expandable primetime type-reg'
        or by using class_ = 'time'

    """
    schedule = defaultdict(dict)
    for year in years:
        yearly_schedule = defaultdict(dict)
        for week in weeks:
            url = f'http://www.nfl.com/schedules/{year}/REG{week}'
            page = requests.get(url)
            soup = BeautifulSoup(page.text, 'html.parser')

            game_info = []
            for comment in soup.findAll(text=lambda text:isinstance(text, Comment)):
                if comment.split()[0] in set(['gameKey.key:', 'awayAbbr:', 'homeAbbr:']):
                    game_info.append(comment.strip().split(':')[-1].strip())

            games = defaultdict(list)
            for info in game_info:
                try:
                    key = pd.to_datetime(info).day_name()
                    games[key]
                except:
                    pass

            for info in game_info:
                try:
                    key = pd.to_datetime(info).day_name()
                    if info in games.keys():
                        key = info
                except:
                    games[key].append(info)

            yearly_schedule[week] = games
        schedule[year] = yearly_schedule

    return schedule

def get_yearly_schedule(years: list, weeks: list):
    """
    Using pro-football reference. Takes returns a dataframe for each year with
    columns of week number, week day of game, date of game, time of game, and
    team playing in game. Therefore, each game will have 2 entries in the DF.
    """
    schedule = defaultdict(dict)
    yearly_schedule = defaultdict(dict)
    for year in years:
        url = f'https://www.pro-football-reference.com/years/'\
              f'{year}/games.htm'
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')

        table = soup.find('table')
        rows = table.findAll('tr')

        res = defaultdict(list)
        stats = ['game_day_of_week', 'game_date', 'gametime']
        for row in rows:
            num = row.find('th', attrs={'scope': 'row'})
            try:
                week_num = int(num.contents[0])
                if week_num in set(weeks):
                    res['week_num'].append(week_num)
                    res['week_num'].append(week_num)

                    for item in row.findAll('td', attrs={'data-stat': stats}):
                        key = item.attrs['data-stat']
                        value = item.contents[0]
                        res[key].append(value)
                        res[key].append(value)

                    teams = ['winner', 'loser']
                    for team in row.findAll('td', attrs={'data-stat': teams}):
                        res['team'].append(team.find('a').contents[0])
            except:
                continue
        df = pd.DataFrame(res)
        gametimes = df['gametime'].values
        times = []
        for gametime in gametimes:
            hour = int(gametime.split(':')[0])
            minute = int(gametime.split(':')[1][:2])
            M = gametime[-2:]
            ts = time(hour+12, minute) if M == 'PM' else time(hour, minute)
            times.append(ts)
        df['gametime'] = times

        yearly_schedule[year] = df
        return yearly_schedule
