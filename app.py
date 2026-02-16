import os
from pathlib import Path
import importlib.util
import json
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from root .env
load_dotenv()

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.offline as pyo
import ast
import re
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import requests
import warnings

# Suppress pandas FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message=".*Series.__getitem__ treating keys as positions.*")

app = Flask(__name__)

# Cached artifacts to avoid repeated heavy loads and to keep routes resilient
role_model: Optional[SentenceTransformer] = None
roles_df: Optional[pd.DataFrame] = None
role_embeddings: Optional[np.ndarray] = None
nn: Optional[NearestNeighbors] = None
role_model_error: Optional[str] = None

diabetes_model = None
diabetes_scaler = None
house_model = None
faq_extractor_module = None

STUDENT_MGMT_DIR = Path(__file__).resolve().parent / "Student_Management"
STUDENT_MGMT_DATA = STUDENT_MGMT_DIR / "students.json"


def ensure_role_data_loaded() -> bool:
    """Lazily load role data and embeddings; keep feature available even if downloads fail."""
    global roles_df, role_model, role_embeddings, nn, role_model_error

    if roles_df is None:
        try:
            roles_df = pd.read_csv('SKILL ADVISORY/roles_catalog_large.csv', quotechar='"', on_bad_lines='skip')
            roles_df.fillna("", inplace=True)
        except Exception as exc:  # broad catch to keep app alive
            role_model_error = f"Could not load roles catalog: {exc}"
            return False

    if role_model is None and role_model_error is None:
        try:
            role_model = SentenceTransformer("all-MiniLM-L6-v2", tokenizer_kwargs={"clean_up_tokenization_spaces": True})
        except Exception as exc:  # pragma: no cover - protects against offline envs
            role_model_error = f"Could not load sentence transformer: {exc}"
            return False

    if role_embeddings is None and role_model_error is None and role_model is not None:
        try:
            role_texts = (roles_df["role_title"] + ". " + roles_df["role_description"]).tolist()
            role_embeddings = role_model.encode(role_texts, convert_to_numpy=True, show_progress_bar=False)
            nn = NearestNeighbors(n_neighbors=min(5, len(role_embeddings)), metric="cosine")
            nn.fit(role_embeddings)
        except Exception as exc:
            role_model_error = f"Could not build embeddings: {exc}"
            return False

    return role_model_error is None


def load_diabetes_artifacts():
    """Load diabetes model and scaler once; stay graceful if files are missing."""
    global diabetes_model, diabetes_scaler
    if diabetes_model is None or diabetes_scaler is None:
        try:
            diabetes_model = pickle.load(open('DIABETES PREDICTION/flask/model.pkl', 'rb'))
            dataset = pd.read_csv('DIABETES PREDICTION/diabetes.csv')
            diabetes_scaler = MinMaxScaler(feature_range=(0, 1))
            diabetes_scaler.fit(dataset.iloc[:, [1, 2, 5, 7]].values)
        except FileNotFoundError:
            diabetes_model = None
            diabetes_scaler = None
    return diabetes_model, diabetes_scaler


def load_house_model():
    """Load house price model lazily; return None if unavailable."""
    global house_model
    if house_model is None:
        try:
            house_model = joblib.load('ADV HOUSE PREDICTION/house_model.pkl')
        except FileNotFoundError:
            house_model = None
    return house_model


def load_faq_extractor():
    """Safely import FAQ extractor helpers from the subfolder with spaces."""
    global faq_extractor_module
    if faq_extractor_module is not None:
        return faq_extractor_module, None

    module_path = Path(__file__).resolve().parent / 'FAQ EXTRACTOR' / 'app.py'
    if not module_path.exists():
        return None, f"FAQ extractor module missing at {module_path}"

    spec = importlib.util.spec_from_file_location('faq_extractor_module', module_path)
    if spec is None or spec.loader is None:
        return None, "Could not load FAQ extractor module specification"

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        faq_extractor_module = module
        return faq_extractor_module, None
    except Exception as exc:  # pragma: no cover - keeps app alive even if parsing fails
        return None, f"Could not import FAQ extractor: {exc}"


def load_student_records():
    if not STUDENT_MGMT_DATA.exists():
        return {}, [], "students.json not found. Add a student to create it."
    try:
        with open(STUDENT_MGMT_DATA, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = sorted(({"name": k, "grade": v} for k, v in data.items()), key=lambda x: x["name"].lower())
        return data, items, None
    except Exception as exc:
        return {}, [], f"Could not read students.json: {exc}"


def save_student_records(data: dict):
    STUDENT_MGMT_DATA.parent.mkdir(parents=True, exist_ok=True)
    with open(STUDENT_MGMT_DATA, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def get_gdp_data():
    DATA_FILENAME = 'GDP DASHBOARD/data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)
    MIN_YEAR = 1960
    MAX_YEAR = 2022
    gdp_df = raw_gdp_df.melt(
        ['Country Code'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'GDP',
    )
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])
    return gdp_df

# IPL Data
matches = pd.read_csv('IPL APP/data/ipl-matches.csv')
balls = pd.read_csv('IPL APP/data/ball.csv')

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

# Helper Functions for IPL
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

# Bowler Data
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

def extract_skills(text, vocab=None):
    if vocab is None:
        vocab = [
            "python", "java", "c++", "react", "node", "django", "flask", "sql",
            "tensorflow", "pytorch", "nlp", "cloud", "aws", "docker", "kubernetes",
            "git", "html", "css", "javascript", "linux", "azure", "pandas", "numpy"
        ]
    text_low = text.lower()
    found = []
    for skill in vocab:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_low):
            found.append(skill)
    return found

def generate_learning_plan(role_title, missing_skills):
    plan = {
        "30 Days": [],
        "60 Days": [],
        "90 Days": []
    }
    if not missing_skills:
        plan["30 Days"].append("Revise existing skills and practice small projects.")
        plan["60 Days"].append("Work on intermediate-level projects in your role domain.")
        plan["90 Days"].append("Prepare for interviews and apply for jobs.")
    else:
        for i, skill in enumerate(missing_skills):
            if i % 3 == 0:
                plan["30 Days"].append(f"Learn basics of {skill} (online tutorials).")
            elif i % 3 == 1:
                plan["60 Days"].append(f"Do a mini-project using {skill}.")
            else:
                plan["90 Days"].append(f"Master {skill} and apply it in a portfolio project.")
    return plan

@app.route('/')
def index():
    return render_template('index.html')

# Diabetes route
@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        model, sc = load_diabetes_artifacts()
        if model is None or sc is None:
            return render_template('diabetes.html', prediction_text="Model artifacts are missing on the server.")

        float_features = [float(x) for x in request.form.values()]
        final_features = [np.array(float_features)]
        prediction = model.predict(sc.transform(final_features))

        if prediction == 1:
            pred = "You have Diabetes, please consult a Doctor."
        else:
            pred = "You don't have Diabetes."
        return render_template('diabetes.html', prediction_text=pred)
    return render_template('diabetes.html')

# GDP Dashboard route
@app.route('/gdp', methods=['GET', 'POST'])
def gdp():
    gdp_df = get_gdp_data()
    min_year = int(gdp_df['Year'].min())
    max_year = int(gdp_df['Year'].max())
    countries = sorted(gdp_df['Country Code'].unique())
    if request.method == 'POST':
        from_year = int(request.form['from_year'])
        to_year = int(request.form['to_year'])
        selected_countries = request.form.getlist('countries')
    else:
        from_year = min_year
        to_year = max_year
        selected_countries = ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN']
    if not selected_countries:
        selected_countries = ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN']
    filtered_gdp_df = gdp_df[
        (gdp_df['Country Code'].isin(selected_countries))
        & (gdp_df['Year'] <= to_year)
        & (from_year <= gdp_df['Year'])
    ]
    # Create line chart
    fig = px.line(filtered_gdp_df, x='Year', y='GDP', color='Country Code', title='GDP over time')
    chart_html = pyo.plot(fig, output_type='div', include_plotlyjs=True)
    # Metrics
    last_year_df = gdp_df[gdp_df['Year'] == to_year]
    first_year_df = gdp_df[gdp_df['Year'] == from_year]
    metrics = []
    for country in selected_countries:
        first_row = first_year_df[first_year_df['Country Code'] == country]
        last_row = last_year_df[last_year_df['Country Code'] == country]
        if not first_row.empty and not last_row.empty:
            first_gdp = first_row['GDP'].values[0] / 1000000000
            last_gdp = last_row['GDP'].values[0] / 1000000000
            if not np.isnan(first_gdp) and first_gdp > 0:
                growth = f'{last_gdp / first_gdp:,.2f}x'
                delta_color = 'normal'
            else:
                growth = 'n/a'
                delta_color = 'off'
            metrics.append({
                'country': country,
                'value': f'{last_gdp:,.0f}B',
                'delta': growth,
                'delta_color': delta_color
            })
        else:
            metrics.append({
                'country': country,
                'value': 'N/A',
                'delta': 'n/a',
                'delta_color': 'off'
            })
    return render_template('gdp.html', chart_html=chart_html, metrics=metrics, from_year=from_year, to_year=to_year, selected_countries=selected_countries, countries=countries, min_year=min_year, max_year=max_year)

# IPL App route
@app.route('/ipl', methods=['GET', 'POST'])
@app.route('/ipl/', methods=['GET', 'POST'])
def ipl():
    teams = sorted(set(matches['Team1']).union(set(matches['Team2'])))
    if request.method == 'POST':
        option = request.form['option']
        result = {}
        if option == "Teams":
            result = teamsAPI()
            result['type'] = 'teams'
        elif option == "Team vs Team":
            t1 = request.form['t1']
            t2 = request.form['t2']
            result = teamVteamAPI(t1, t2)
            result['type'] = 'team_vs_team'
            result['t1'] = t1
            result['t2'] = t2
        elif option == "Team Record":
            team = request.form['team']
            result = team_recordAPI(team)
            result['type'] = 'team_record'
            result['team'] = team
        elif option == "Batsman Stats":
            batsman = request.form['batsman']
            data = batsmanAPI(batsman)
            result = {'type': 'batsman', 'data': data, 'batsman': batsman}
        elif option == "Bowler Stats":
            bowler = request.form['bowler']
            data = bowlerAPI(bowler)
            result = {'type': 'bowler', 'data': data, 'bowler': bowler}
        return render_template('ipl.html', result=result, all_players=all_players, teams=teams)
    else:
        return render_template('ipl.html', all_players=all_players, teams=teams)

# Weather App route
@app.route('/weather', methods=['GET', 'POST'])
def weather():
    if request.method == 'POST':
        city = request.form['city']
        weather_key = os.getenv("WEATHER_API_KEY")
        if not weather_key:
            return render_template('weather.html', error="API key not found.")
        url = 'https://api.openweathermap.org/data/2.5/weather'
        params = {'APPID': weather_key, 'q': city, 'units': 'metric'}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            weather_data = response.json()
            city_name = weather_data['name']
            conditions = weather_data['weather'][0]['description'].capitalize()
            temp = weather_data['main']['temp']
            icon = weather_data['weather'][0]['icon']
            weather_info = {
                'city': city_name,
                'conditions': conditions,
                'temp': temp,
                'icon': f'/static/weather_icons/{icon}.png'  # Assuming icons are served from static
            }
            return render_template('weather.html', weather=weather_info)
        except requests.exceptions.RequestException as e:
            return render_template('weather.html', error=f"Error fetching weather: {e}")
    return render_template('weather.html')

# Skill Advisory route
@app.route('/skill', methods=['GET', 'POST'])
def skill():
    if request.method == 'POST':
        if not ensure_role_data_loaded():
            return render_template('skill.html', error=role_model_error or "Skill advisory model is unavailable right now.")

        input_type = request.form.get('input_type')
        if input_type == 'Paste Resume':
            resume_text = request.form.get('resume_text', '')
            user_skills = extract_skills(resume_text)
        elif input_type == 'Enter Skills':
            skills_text = request.form.get('skills', '')
            user_skills = [s.strip() for s in skills_text.split(',') if s.strip()]
        else:
            user_skills = []

        if not user_skills:
            return render_template('skill.html', error="Please provide resume text or skills.")

        skill_sentence = ", ".join(user_skills)
        user_emb = role_model.encode([skill_sentence], convert_to_numpy=True)

        distances, idxs = nn.kneighbors(user_emb, n_neighbors=min(5, len(role_embeddings)))
        recommendations = []
        for dist, idx in zip(distances[0], idxs[0]):
            score = 1 - float(dist)
            role = roles_df.iloc[idx]
            required = [s.strip().lower() for s in str(role.get("required_skills", "")).split(",") if s.strip()]
            missing = [s for s in required if s not in [x.lower().strip() for x in user_skills]]

            plan = generate_learning_plan(role["role_title"], missing)
            recommendations.append({
                'title': role['role_title'],
                'description': role['role_description'],
                'score': round(score, 3),
                'missing_skills': missing,
                'plan': plan
            })

        return render_template('skill.html', recommendations=recommendations, user_skills=user_skills)
    return render_template('skill.html')

# India Census route
@app.route('/census', methods=['GET', 'POST'])
def census():
    df = pd.read_csv('INDIA CENSUS/india.csv')
    states = sorted(df['State'].unique())
    selected_state = request.form.get('state', 'Overall INDIA') if request.method == 'POST' else 'Overall INDIA'
    if selected_state == 'Overall INDIA':
        state_data = df.groupby('State').agg({'Population': 'sum', 'Latitude': 'mean', 'Longitude': 'mean'}).reset_index()
        import folium
        folium_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
        for _, row in state_data.iterrows():
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=f"{row['State']}: {row['Population']:,}"
            ).add_to(folium_map)
        map_html = folium_map._repr_html_()
        filtered_df = df.head(20)  # Overall, show top 20 districts
    else:
        filtered_df = df[df['State'] == selected_state]
        filtered_df = filtered_df.head(20)
        import folium
        folium_map = folium.Map(location=[filtered_df['Latitude'].mean(), filtered_df['Longitude'].mean()], zoom_start=7)
        for _, row in filtered_df.iterrows():
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=f"{row['District']}: {row['Population']:,}"
            ).add_to(folium_map)
        map_html = folium_map._repr_html_()
    return render_template('census.html', df=filtered_df.to_html(index=False), states=states, selected_state=selected_state, map_html=map_html)

# Attendance System route
@app.route('/attendance')
def attendance():
    msg = (
        "Use the StudentAttendance module (register, train, then mark_attendance) from the StudentAttendance folder. "
        "This page provides the Streamlit link and local-camera preview only."
    )
    return render_template('attendance.html', message=msg)


@app.route('/students', methods=['GET', 'POST'])
def student_management():
    data, records, error = load_student_records()
    message = None

    if request.method == 'POST':
        action = request.form.get('action')
        name = (request.form.get('name') or '').strip()

        if action == 'add':
            grade_raw = request.form.get('grade')
            if not name or grade_raw is None or grade_raw == '':
                message = "Name and grade are required."
            else:
                try:
                    grade = float(grade_raw)
                    data[name] = grade
                    save_student_records(data)
                    message = f"Saved {name}."
                except ValueError:
                    message = "Grade must be a number."
        elif action == 'delete':
            if name in data:
                data.pop(name, None)
                save_student_records(data)
                message = f"Deleted {name}."
            else:
                message = f"{name} not found."

        data, records, error = load_student_records()

    return render_template(
        'student_management.html',
        records=records,
        error=error,
        message=message,
        data_file=str(STUDENT_MGMT_DATA),
        streamlit_cmd="streamlit run Student_Management/stud_managementSTREAMLIT.py",
        tkinter_cmd="python Student_Management/stud_managementTK.py",
    )

# House Prediction route
@app.route('/house', methods=['GET', 'POST'])
def house():
    if request.method == 'POST':
        # Get form data
        lot_area = float(request.form['lot_area'])
        year_built = int(request.form['year_built'])
        first_flr_sf = float(request.form['first_flr_sf'])
        second_flr_sf = float(request.form['second_flr_sf'])
        full_bath = int(request.form['full_bath'])
        bedroom_abv_gr = int(request.form['bedroom_abv_gr'])
        tot_rms_abv_grd = int(request.form['tot_rms_abv_grd'])
        overall_qual = int(request.form['overall_qual'])
        overall_cond = int(request.form['overall_cond'])

        # Create DataFrame for input
        input_data = pd.DataFrame({
            'LotArea': [lot_area],
            'YearBuilt': [year_built],
            '1stFlrSF': [first_flr_sf],
            '2ndFlrSF': [second_flr_sf],
            'FullBath': [full_bath],
            'BedroomAbvGr': [bedroom_abv_gr],
            'TotRmsAbvGrd': [tot_rms_abv_grd],
            'OverallQual': [overall_qual],
            'OverallCond': [overall_cond]
        })

        # Handle missing values (though unlikely)
        imputer = SimpleImputer(strategy='median')
        input_data = pd.DataFrame(imputer.fit_transform(input_data), columns=input_data.columns)

        # Load model once and reuse
        model = load_house_model()
        if model is None:
            return render_template('house.html', prediction="Model file is missing on the server.")

        # Predict
        prediction = model.predict(input_data)[0]

        return render_template('house.html', prediction=round(prediction, 2))

    return render_template('house.html')

# SQL Comparison route
@app.route('/sql', methods=['GET', 'POST'])
def sql():
    if request.method == 'POST':
        # Assume files are db1.sql and db2.sql in SQL COMPARISION
        import subprocess
        try:
            subprocess.run(['python', 'compare_sql.py'], cwd='SQL COMPARISION', check=True)
            # Read the summary
            summary_df = pd.read_csv('SQL COMPARISION/summary/db_comparison_summary.csv')
            report_df = pd.read_csv('SQL COMPARISION/reports/db_comparison_report.csv')
            return render_template('sql.html', summary=summary_df.to_html(), report=report_df.to_html())
        except subprocess.CalledProcessError as e:
            return render_template('sql.html', error=f"Error running comparison: {e}")
    return render_template('sql.html', message="Click 'Compare' to run SQL comparison on db1.sql and db2.sql.")

# FAQ Extractor route
@app.route('/faq', methods=['GET', 'POST'])
def faq():
    if request.method == 'POST':
        url = request.form.get('url')
        if not url:
            return render_template('faq.html', error="Please provide a URL.")
        try:
            extractor, err = load_faq_extractor()
            if err:
                return render_template('faq.html', error=err)

            html = extractor.fetch_url(url)
            faqs = extractor.extract_faqs_from_html(html)
            return render_template('faq.html', faqs=faqs[:10])  # Limit to 10
        except Exception as e:
            return render_template('faq.html', error=f"Error extracting FAQs: {e}")
    return render_template('faq.html', message="Enter a URL to extract FAQs from websites.")

if __name__ == "__main__":
    app.run(debug=True)