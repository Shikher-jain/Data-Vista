import pandas as pd
import numpy as np

# Load data
matches = pd.read_csv("data/ipl-matches.csv")
balls = pd.read_csv("data/ball.csv")

# Merge ball-by-ball data with match info
ball_withmatch = balls.merge(matches, on='ID', how='inner').copy()
ball_withmatch['BowlingTeam'] = ball_withmatch.Team1 + ball_withmatch.Team2
ball_withmatch['BowlingTeam'] = ball_withmatch[['BowlingTeam', 'BattingTeam']].apply(lambda x: x.values[0].replace(x.values[1], ''), axis=1)
batter_data = ball_withmatch[np.append(balls.columns.values, ['BowlingTeam', 'Player_of_Match'])]

# Utility function for conversion
def convert(obj):
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(i) for i in obj]
    else:
        return obj

# Teams API
def teamsAPI():
    teams = sorted(set(matches['Team1']).union(set(matches['Team2'])))
    return {'teams': teams}

# Team vs Team API
def teamVteamAPI(T1, T2):
    valid_teams = set(matches['Team1']).union(set(matches['Team2']))
    if (T1 in valid_teams) and (T2 in valid_teams):
        temp_df = matches[((matches['Team1'] == T1) & (matches['Team2'] == T2)) |
                          ((matches['Team1'] == T2) & (matches['Team2'] == T1))]
        total_matches = temp_df.shape[0]
        win_counts = temp_df['WinningTeam'].value_counts()
        matches_won_T1 = win_counts.get(T1, 0)
        matches_won_T2 = win_counts.get(T2, 0)
        draws = total_matches - (matches_won_T1 + matches_won_T2)
        return {
            'total_matches': convert(total_matches),
            T1: convert(matches_won_T1),
            T2: convert(matches_won_T2),
            'draws': convert(draws)
        }
    else:
        return {"Message": 'Invalid Team !!'}

# Team overall stats
def allRound(team):
    df = matches[(matches['Team1'] == team) | (matches['Team2'] == team)].copy()
    mp = df.shape[0]
    won = df[df.WinningTeam == team].shape[0]
    nr = df[df.WinningTeam.isnull()].shape[0]
    loss = mp - nr - won
    nt = df[(df.MatchNumber == 'Final') & (df.WinningTeam == team)].shape[0]
    win_rate = round(won / mp * 100, 2) if mp else 0
    return {
        'Matches Played': mp,
        'Wins': won,
        'Losses': loss,
        'No Result': nr,
        'Titles': nt,
        'Win Rate': win_rate
    }

# Team record API
def team_recordAPI(team):
    self_record = allRound(team)
    TEAMS = sorted(set(matches.Team1).union(set(matches.Team2)))
    against = {team2: teamVteamAPI(team, team2) for team2 in TEAMS if team2 != team}
    data = {team: {'Overall': convert(self_record), 'Against': convert(against)}}
    return data

# Batsman stats
def batsmanrecord(name, df):
    if df.empty:
        return np.nan
    out = df[df.player_out == name].shape[0]
    df = df[df['batter'] == name]
    inngs = df.ID.unique().shape[0]
    runs = df.batsman_run.sum()
    fours = df[(df.batsman_run == 4) & (df.non_boundary == 0)].shape[0]
    sixes = df[(df.batsman_run == 6) & (df.non_boundary == 0)].shape[0]
    avg = runs / out if out else np.inf
    nballs = df[~df.extra_type.isin(['wides', 'noballs'])].shape[0]
    strike_rate = (runs / nballs * 100) if nballs else 0
    gb = df.groupby('ID').sum()
    fiftes = gb[(gb.batsman_run >= 50) & (gb.batsman_run < 100)].shape[0]
    hundreds = gb[gb.batsman_run >= 100].shape[0]
    highest_score = gb.batsman_run.max()
    not_out = inngs - out
    mom = df[df.Player_of_Match == name].drop_duplicates('ID', keep='first').shape[0]
    data = {
        'Innings': inngs,
        'Runs': runs,
        'Fours': fours,
        'Sixes': sixes,
        'Average': avg,
        'Strike Rate': strike_rate,
        'Fifties': fiftes,
        'Hundreds': hundreds,
        'High Score': highest_score,
        'Not Out': not_out,
        'Man Of The Match': mom
    }
    return data


# Bowler stats

def bowlerRun(x):
    return 0 if x.iloc[0] in ['penalty', 'legbyes', 'byes'] else x.iloc[1]

def bowlerWicket(x):
    return x.iloc[1] if x.iloc[0] in ['caught', 'caught and bowled', 'bowled', 'stumped', 'lbw', 'hit wicket'] else 0

bowler_data = batter_data.copy()
bowler_data['bowler_run'] = bowler_data[['extra_type', 'total_run']].apply(bowlerRun, axis=1)
bowler_data['isBowlerWicket'] = bowler_data[['kind', 'isWicketDelivery']].apply(bowlerWicket, axis=1)

def bowlerRecord(bowler, df):
    df = df[df['bowler'] == bowler]
    inngs = df.ID.unique().shape[0]
    nballs = df[~df.extra_type.isin(['wides', 'noballs'])].shape[0]
    runs = df['bowler_run'].sum()
    eco = (runs / nballs * 6) if nballs else 0
    wicket = df.isBowlerWicket.sum()
    avg = runs / wicket if wicket else np.inf
    strike_rate = (nballs / wicket) if wicket else np.nan
    gb = df.groupby('ID').sum()
    w3 = gb[gb.isBowlerWicket >= 3].shape[0]
    best_wicket = gb.sort_values(['isBowlerWicket', 'bowler_run'], ascending=[False, True])[['isBowlerWicket', 'bowler_run']].head(1).values
    best_figure = f'{best_wicket[0][0]}/{best_wicket[0][1]}' if best_wicket.size > 0 else np.nan
    mom = df[df.Player_of_Match == bowler].drop_duplicates('ID', keep='first').shape[0]
    return {
        'innings': inngs, 'wicket': wicket, 'economy': eco, 'average': avg,
        'strikeRate': strike_rate, 'fours': df[(df.batsman_run == 4) & (df.non_boundary == 0)].shape[0],
        'sixes': df[(df.batsman_run == 6) & (df.non_boundary == 0)].shape[0], 'best_figure': best_figure, '3+W': w3, 'mom': mom
    }

def bowlerAPI(bowler, balls=bowler_data):
    df = balls[balls.innings.isin([1, 2])]
    self_record = bowlerRecord(bowler, df=df)
    TEAMS = sorted(set(matches.Team1).union(set(matches.Team2)))
    against = {team: bowlerRecord(bowler, df[df.BattingTeam == team]) for team in TEAMS}
    return {bowler: {'all': convert(self_record), 'Against': convert(against)}}
