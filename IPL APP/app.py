import streamlit as st

import pandas as pd
import numpy as np
import ast

st.set_page_config(page_title="IPL Stats Dashboard", layout="wide")
# ------------------ Data Load ------------------
matches = pd.read_csv("data/ipl-matches.csv")
balls = pd.read_csv("data/ball.csv")

# Safely get all unique players from the 'Team1Players' column
s = matches["Team1Players"].sum()
parts = s.split("][")
parts = [p if p.startswith("[") else "[" + p for p in parts]
parts = [p if p.endswith("]") else p + "]" for p in parts]
lists = [ast.literal_eval(p) for p in parts]
all_players = sorted(set(sum(lists, [])))

# Merge ball-by-ball data with match info for detailed analysis
ball_withmatch = balls.merge(matches, on='ID', how='inner').copy()
ball_withmatch['BowlingTeam'] = ball_withmatch.Team1 + ball_withmatch.Team2
ball_withmatch['BowlingTeam'] = ball_withmatch[['BowlingTeam', 'BattingTeam']].apply(lambda x: x.values[0].replace(x.values[1], ''), axis=1)
batter_data = ball_withmatch[np.append(balls.columns.values, ['BowlingTeam', 'Player_of_Match'])]

# ------------------ Helper Functions ------------------
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

def teamsAPI():
    teams = sorted(set(matches['Team1']).union(set(matches['Team2'])))
    return {'teams': teams}

def teamVteamAPI(T1, T2):
    valid_teams = sorted(set(matches['Team1']).union(set(matches['Team2'])))
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

def allRound(team):
    df = matches[(matches['Team1'] == team) | (matches['Team2'] == team)].copy()
    mp = df.shape[0]
    won = df[df.WinningTeam == team].shape[0]
    nr = df[df.WinningTeam.isnull()].shape[0]
    loss = mp - nr - won
    nt = df[(df.MatchNumber == 'Final') & (df.WinningTeam == team)].shape[0]
    win_rate = round(won / mp * 100, 2) if mp > 0 else 0

    return {
        'Matches Played': mp,
        'Won': won,
        'Loss': loss,
        'noResult': nr,
        'title': nt,
        'Win Rate': win_rate
    }

def team_recordAPI(team):
    self_record = allRound(team)
    TEAMS = sorted(set(matches['Team1']).union(set(matches['Team2'])))
    against = {team2: teamVteamAPI(team, team2) for team2 in TEAMS if team2 != team}
    data = {team: {'Overall': convert(self_record),
                   'Against': convert(against)}}
    return data

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

    if not gb.empty:
        highest_score = gb.batsman_run.max()
        highest_score_id = gb.batsman_run.idxmax()
        is_out = df[(df.ID == highest_score_id) & (df.player_out == name)].shape[0] > 0
        if not is_out:
            highest_score = str(highest_score) + "*"
        else:
            highest_score = str(highest_score)
    else:
        highest_score = np.nan

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

def batsmanVsTeam(batsman, team, df):
    df = df[df.BowlingTeam == team].copy()
    return batsmanrecord(batsman, df)

def batsmanAPI(name, balls=batter_data):
    df = balls[balls.innings.isin([1, 2])]
    self_record = batsmanrecord(name, df=df)
    TEAMS = sorted(set(matches['Team1']).union(set(matches['Team2'])))
    against = {team: batsmanVsTeam(name, team, df) for team in TEAMS}
    data = {name: {'all': convert(self_record),
                   "Against": convert(against)}}
    return data

# ------------------ Bowler Data ------------------
bowler_data = batter_data.copy()

def bowlerRun(x):
    if x[0] in ['penalty', 'legbyes', 'byes']:
        return 0
    else:
        return x[1]

bowler_data['bowler_run'] = bowler_data[['extra_type', 'total_run']].apply(bowlerRun, axis=1)

def bowlerWicket(x):
    if x[0] in ['caught', 'caught and bowled', 'bowled', 'stumped', 'lbw', 'hit wicket']:
        return x[1]
    else:
        return 0

bowler_data['isBowlerWicket'] = bowler_data[['kind', 'isWicketDelivery']].apply(bowlerWicket, axis=1)

def bowlerRecord(bowler, df):
    df = df[df['bowler'] == bowler]
    inngs = df.ID.unique().shape[0]
    nballs = df[~(df.extra_type.isin(['wides', 'noballs']))].shape[0]
    runs = df['bowler_run'].sum()
    eco = runs / nballs * 6 if nballs else 0
    wicket = df.isBowlerWicket.sum()
    avg = runs / wicket if wicket else np.inf
    strike_rate = nballs / wicket if wicket else np.nan
    gb = df.groupby('ID').sum()
    w3 = gb[(gb.isBowlerWicket >= 3)].shape[0]
    best_wicket = gb.sort_values(['isBowlerWicket', 'bowler_run'], ascending=[False, True])[['isBowlerWicket', 'bowler_run']].head(1).values
    best_figure = f'{best_wicket[0][0]}/{best_wicket[0][1]}' if best_wicket.size > 0 else np.nan
    mom = df[df.Player_of_Match == bowler].drop_duplicates('ID', keep='first').shape[0]
    data = {
        'innings': inngs,
        'wicket': wicket,
        'economy': eco,
        'average': avg,
        'strikeRate': strike_rate,
        'fours': df[(df.batsman_run == 4) & (df.non_boundary == 0)].shape[0],
        'sixes': df[(df.batsman_run == 6) & (df.non_boundary == 0)].shape[0],
        'best_figure': best_figure,
        '3+W': w3,
        'mom': mom
    }
    return data

def bowlerVsTeam(bowler, team, df):
    df = df[df.BattingTeam == team].copy()
    return bowlerRecord(bowler, df)

def bowlerAPI(bowler, balls=bowler_data):
    df = balls[balls.innings.isin([1, 2])]
    self_record = bowlerRecord(bowler, df=df)
    TEAMS = sorted(set(matches['Team1']).union(set(matches['Team2'])))
    against = {team: bowlerVsTeam(bowler, team, df) for team in TEAMS}
    data = {bowler: {'all': convert(self_record), 'Against': convert(against)}}
    return data

# ------------------ Streamlit UI ------------------
st.title("🏏 IPL Stats Dashboard")

option = st.sidebar.selectbox(
    "Choose Analysis",
    ["Teams", "Team vs Team", "Team Record", "Batsman Stats", "Bowler Stats"]
)

# --- Teams ---
if option == "Teams":
    st.header("All IPL Teams")
    teams = teamsAPI()
    for n, team in enumerate(teams['teams'], start=1):
        st.write(f"{n}. {team}")
    if st.button("Get Teams Data in JSON"):
        st.json(teams)

# --- Team vs Team ---
elif option == "Team vs Team":
    st.header("Team vs Team Analysis")
    valid_teams = sorted(set(matches['Team1']).union(set(matches['Team2'])))
    t1 = st.selectbox("Select Team 1", valid_teams)
    t2 = st.selectbox("Select Team 2", valid_teams)
    if st.button("Compare"):
        result = teamVteamAPI(t1, t2)
        st.subheader(f"Total Matches: {result['total_matches']}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write(f"**{t1} Wins** : {result[t1]}")
        with c2:
            st.write(f"**{t2} Wins** : {result[t2]}")
        with c3:
            st.write(f"**Draws** : {result['draws']}")
        chart_data = pd.DataFrame({
            "Team": [t1, t2, "Draws"],
            "Matches": [result[t1], result[t2], result["draws"]]
        })
        st.bar_chart(chart_data.set_index("Team"))
        if st.button("Get Team vs Team Data in JSON"):
            st.json(result)

# --- Team Record ---
elif option == "Team Record":
    st.header("Overall + Against Stats")
    valid_teams = sorted(set(matches['Team1']).union(set(matches['Team2'])))
    team = st.selectbox("Select Team", valid_teams)
    if st.button("Get Record"):
        record = team_recordAPI(team)
        st.subheader(f"Overall Record for {team}")
        st.json(record[team]['Overall'])
        st.subheader(f"Team Records Against Other Teams")
        st.json(record[team]['Against'])

# --- Batsman Stats ---
elif option == "Batsman Stats":
    st.header("Batsman Performance")
    batsman = st.selectbox("Select Player", all_players)
    if st.button("Get Batsman Stats"):
        data = batsmanAPI(batsman)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Runs", data[batsman]['all']['Runs'])
        with col2:
            st.metric("Innings", data[batsman]['all']['Innings'])
        with col3:
            st.metric("Average", f"{data[batsman]['all']['Average']:.2f}" if data[batsman]['all']['Average'] != float('inf') else "∞")
        with col4:
            st.metric("Strike Rate", f"{data[batsman]['all']['Strike Rate']:.2f}")
        with col5:
            st.metric("Highest Score", data[batsman]['all']['High Score'])
        st.subheader("Full Batsman Data")
        st.json(data)

# --- Bowler Stats ---
elif option == "Bowler Stats":
    st.header("Bowler Performance")
    bowler = st.selectbox("Select Player", all_players)
    if st.button("Get Bowler Stats"):
        data = bowlerAPI(bowler)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Wickets", data[bowler]['all']['wicket'])
        with col2:
            st.metric("Innings", data[bowler]['all']['innings'])
        with col3:
            st.metric("Economy", f"{data[bowler]['all']['economy']:.2f}")
        with col4:
            st.metric("Average", f"{data[bowler]['all']['average']:.2f}" if data[bowler]['all']['average'] != float('inf') else "∞")
        with col5:
            st.metric("Strike Rate", f"{data[bowler]['all']['strikeRate']:.2f}" if not pd.isna(data[bowler]['all']['strikeRate']) else "N/A")
        st.subheader("Full Bowler Data")
        st.json(data)
