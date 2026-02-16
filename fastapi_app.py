import os
import sys
import ast
import re
import subprocess
import warnings
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.offline as pyo
import requests
from sentence_transformers import SentenceTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=FutureWarning, message=".*Series.__getitem__ treating keys as positions.*")
load_dotenv()

app = FastAPI(title="Data Vista (FastAPI)")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def get_gdp_data() -> pd.DataFrame:
    data_filename = "GDP DASHBOARD/data/gdp_data.csv"
    raw_gdp_df = pd.read_csv(data_filename)
    min_year = 1960
    max_year = 2022
    gdp_df = raw_gdp_df.melt(
        ["Country Code"],
        [str(x) for x in range(min_year, max_year + 1)],
        "Year",
        "GDP",
    )
    gdp_df["Year"] = pd.to_numeric(gdp_df["Year"])
    return gdp_df


matches = pd.read_csv("IPL APP/data/ipl-matches.csv")
balls = pd.read_csv("IPL APP/data/ball.csv")

s = matches["Team1Players"].sum()
parts = s.split("][")
parts = [p if p.startswith("[") else "[" + p for p in parts]
parts = [p if p.endswith("]") else p + "]" for p in parts]
lists = [ast.literal_eval(p) for p in parts]
all_players = sorted(set(sum(lists, [])))

ball_withmatch = balls.merge(matches, on="ID", how="inner").copy()
ball_withmatch["BowlingTeam"] = ball_withmatch.Team1 + ball_withmatch.Team2
ball_withmatch["BowlingTeam"] = ball_withmatch[["BowlingTeam", "BattingTeam"]].apply(
    lambda x: x.values[0].replace(x.values[1], ""), axis=1
)
batter_data = ball_withmatch[np.append(balls.columns.values, ["BowlingTeam", "Player_of_Match"])]


def convert(obj):
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert(i) for i in obj]
    return obj


def teams_api():
    teams = sorted(set(matches["Team1"]).union(set(matches["Team2"])))
    return {"teams": teams}


def team_v_team_api(t1: str, t2: str):
    valid_teams = sorted(set(matches["Team1"]).union(set(matches["Team2"])))
    if t1 in valid_teams and t2 in valid_teams:
        temp_df = matches[
            ((matches["Team1"] == t1) & (matches["Team2"] == t2))
            | ((matches["Team1"] == t2) & (matches["Team2"] == t1))
        ]
        total_matches = temp_df.shape[0]
        win_counts = temp_df["WinningTeam"].value_counts()
        matches_won_t1 = win_counts.get(t1, 0)
        matches_won_t2 = win_counts.get(t2, 0)
        draws = total_matches - (matches_won_t1 + matches_won_t2)
        return {
            "total_matches": convert(total_matches),
            t1: convert(matches_won_t1),
            t2: convert(matches_won_t2),
            "draws": convert(draws),
        }
    return {"Message": "Invalid Team !!"}


def all_round(team: str):
    df = matches[(matches["Team1"] == team) | (matches["Team2"] == team)].copy()
    mp = df.shape[0]
    won = df[df.WinningTeam == team].shape[0]
    nr = df[df.WinningTeam.isnull()].shape[0]
    loss = mp - nr - won
    nt = df[(df.MatchNumber == "Final") & (df.WinningTeam == team)].shape[0]
    win_rate = round(won / mp * 100, 2) if mp > 0 else 0
    return {
        "Matches Played": mp,
        "Won": won,
        "Loss": loss,
        "noResult": nr,
        "title": nt,
        "Win Rate": win_rate,
    }


def team_record_api(team: str):
    self_record = all_round(team)
    teams = sorted(set(matches["Team1"]).union(set(matches["Team2"])))
    against = {team2: team_v_team_api(team, team2) for team2 in teams if team2 != team}
    data = {team: {"Overall": convert(self_record), "Against": convert(against)}}
    return data


def batsmanrecord(name: str, df: pd.DataFrame):
    if df.empty:
        return np.nan
    out = df[df.player_out == name].shape[0]
    df = df[df["batter"] == name]
    inngs = df.ID.unique().shape[0]
    runs = df.batsman_run.sum()
    fours = df[(df.batsman_run == 4) & (df.non_boundary == 0)].shape[0]
    sixes = df[(df.batsman_run == 6) & (df.non_boundary == 0)].shape[0]
    avg = runs / out if out else np.inf
    nballs = df[~df.extra_type.isin(["wides", "noballs"])].shape[0]
    strike_rate = (runs / nballs * 100) if nballs else 0
    gb = df.groupby("ID").sum()
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
    mom = df[df.Player_of_Match == name].drop_duplicates("ID", keep="first").shape[0]
    data = {
        "Innings": inngs,
        "Runs": runs,
        "Fours": fours,
        "Sixes": sixes,
        "Average": avg,
        "Strike Rate": strike_rate,
        "Fifties": fiftes,
        "Hundreds": hundreds,
        "High Score": highest_score,
        "Not Out": not_out,
        "Man Of The Match": mom,
    }
    return data


def batsman_vs_team(batsman: str, team: str, df: pd.DataFrame):
    df = df[df.BowlingTeam == team].copy()
    return batsmanrecord(batsman, df)


def batsman_api(name: str, balls: pd.DataFrame = batter_data):
    df = balls[balls.innings.isin([1, 2])]
    self_record = batsmanrecord(name, df=df)
    teams = sorted(set(matches["Team1"]).union(set(matches["Team2"])))
    against = {team: batsman_vs_team(name, team, df) for team in teams}
    data = {name: {"all": convert(self_record), "Against": convert(against)}}
    return data


bowler_data = batter_data.copy()


def bowler_run(x):
    if x[0] in ["penalty", "legbyes", "byes"]:
        return 0
    return x[1]


def bowler_wicket(x):
    if x[0] in ["caught", "caught and bowled", "bowled", "stumped", "lbw", "hit wicket"]:
        return x[1]
    return 0


bowler_data["bowler_run"] = bowler_data[["extra_type", "total_run"]].apply(bowler_run, axis=1)
bowler_data["isBowlerWicket"] = bowler_data[["kind", "isWicketDelivery"]].apply(bowler_wicket, axis=1)


def bowler_record(bowler: str, df: pd.DataFrame):
    df = df[df["bowler"] == bowler]
    inngs = df.ID.unique().shape[0]
    nballs = df[~(df.extra_type.isin(["wides", "noballs"]))].shape[0]
    runs = df["bowler_run"].sum()
    eco = runs / nballs * 6 if nballs else 0
    wicket = df.isBowlerWicket.sum()
    avg = runs / wicket if wicket else np.inf
    strike_rate = nballs / wicket if wicket else np.nan
    gb = df.groupby("ID").sum()
    w3 = gb[(gb.isBowlerWicket >= 3)].shape[0]
    best_wicket_df = (
        gb.sort_values(["isBowlerWicket", "bowler_run"], ascending=[False, True])[["isBowlerWicket", "bowler_run"]]
        .head(1)
        .values
    )
    best_figure = f"{best_wicket_df[0][0]}/{best_wicket_df[0][1]}" if best_wicket_df.size > 0 else np.nan
    mom = df[df.Player_of_Match == bowler].drop_duplicates("ID", keep="first").shape[0]
    data = {
        "innings": inngs,
        "wicket": wicket,
        "economy": eco,
        "average": avg,
        "strikeRate": strike_rate,
        "fours": df[(df.batsman_run == 4) & (df.non_boundary == 0)].shape[0],
        "sixes": df[(df.batsman_run == 6) & (df.non_boundary == 0)].shape[0],
        "best_figure": best_figure,
        "3+W": w3,
        "mom": mom,
    }
    return data


def bowler_vs_team(bowler: str, team: str, df: pd.DataFrame):
    df = df[df.BattingTeam == team].copy()
    return bowler_record(bowler, df)


def bowler_api(bowler: str, balls: pd.DataFrame = bowler_data):
    df = balls[balls.innings.isin([1, 2])]
    self_record = bowler_record(bowler, df=df)
    teams = sorted(set(matches["Team1"]).union(set(matches["Team2"])))
    against = {team: bowler_vs_team(bowler, team, df) for team in teams}
    data = {bowler: {"all": convert(self_record), "Against": convert(against)}}
    return data


role_model = None
role_embeddings = None
roles_df = None
nn = None
role_model_error = None


def ensure_role_data_loaded():
    """Lazily load role data and embeddings; stay resilient when offline."""
    global roles_df, role_model, role_embeddings, nn, role_model_error
    if roles_df is None:
        try:
            roles_df = pd.read_csv("SKILL ADVISORY/roles_catalog_large.csv", quotechar="\"", on_bad_lines="skip")
            roles_df.fillna("", inplace=True)
        except Exception as exc:  # pylint: disable=broad-except
            role_model_error = f"Could not load roles catalog: {exc}"
            return False
    if role_model is None and role_model_error is None:
        try:
            role_model = SentenceTransformer(
                "all-MiniLM-L6-v2", tokenizer_kwargs={"clean_up_tokenization_spaces": True}
            )
        except Exception as exc:  # pylint: disable=broad-except
            role_model_error = f"Could not load sentence transformer: {exc}"
            return False
    if role_embeddings is None and role_model_error is None and role_model is not None:
        try:
            role_texts = (roles_df["role_title"] + ". " + roles_df["role_description"]).tolist()
            role_embeddings = role_model.encode(role_texts, convert_to_numpy=True, show_progress_bar=False)
            nn = NearestNeighbors(n_neighbors=min(5, len(role_embeddings)), metric="cosine")
            nn.fit(role_embeddings)
        except Exception as exc:  # pylint: disable=broad-except
            role_model_error = f"Could not build embeddings: {exc}"
            return False
    return role_model_error is None


def extract_skills(text: str, vocab: Optional[List[str]] = None) -> List[str]:
    if vocab is None:
        vocab = [
            "python",
            "java",
            "c++",
            "react",
            "node",
            "django",
            "flask",
            "sql",
            "tensorflow",
            "pytorch",
            "nlp",
            "cloud",
            "aws",
            "docker",
            "kubernetes",
            "git",
            "html",
            "css",
            "javascript",
            "linux",
            "azure",
            "pandas",
            "numpy",
        ]
    text_low = text.lower()
    found = []
    for skill in vocab:
        pattern = r"\\b" + re.escape(skill.lower()) + r"\\b"
        if re.search(pattern, text_low):
            found.append(skill)
    return found


def generate_learning_plan(role_title: str, missing_skills: List[str]):
    plan = {"30 Days": [], "60 Days": [], "90 Days": []}
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


try:
    diabetes_model = pickle.load(open("DIABETES PREDICTION/flask/model.pkl", "rb"))
    diabetes_dataset = pd.read_csv("DIABETES PREDICTION/diabetes.csv")
    diabetes_scaler = MinMaxScaler(feature_range=(0, 1))
    diabetes_scaler.fit(diabetes_dataset.iloc[:, [1, 2, 5, 7]].values)
except FileNotFoundError:
    diabetes_model = None
    diabetes_scaler = None

try:
    house_model = joblib.load("ADV HOUSE PREDICTION/house_model.pkl")
except FileNotFoundError:
    house_model = None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/diabetes", response_class=HTMLResponse)
async def diabetes_get(request: Request):
    return templates.TemplateResponse("diabetes.html", {"request": request})


@app.post("/diabetes", response_class=HTMLResponse)
async def diabetes_post(
    request: Request,
    glucose: float = Form(...),
    bloodpressure: float = Form(...),
    insulin: float = Form(...),
    bmi: float = Form(...),
):
    if not diabetes_model or not diabetes_scaler:
        return templates.TemplateResponse(
            "diabetes.html",
            {"request": request, "prediction_text": "Model is missing on server."},
        )
    final_features = np.array([[glucose, bloodpressure, insulin, bmi]])
    prediction = diabetes_model.predict(diabetes_scaler.transform(final_features))
    pred_text = (
        "You have Diabetes, please consult a Doctor." if prediction == 1 else "You don't have Diabetes."
    )
    return templates.TemplateResponse(
        "diabetes.html", {"request": request, "prediction_text": pred_text}
    )


def _build_gdp_context(from_year: Optional[int], to_year: Optional[int], selected_countries: Optional[List[str]]):
    gdp_df = get_gdp_data()
    min_year = int(gdp_df["Year"].min())
    max_year = int(gdp_df["Year"].max())
    countries = sorted(gdp_df["Country Code"].unique())
    if from_year is None:
        from_year = min_year
    if to_year is None:
        to_year = max_year
    if not selected_countries:
        selected_countries = ["DEU", "FRA", "GBR", "BRA", "MEX", "JPN"]
    filtered_gdp_df = gdp_df[
        (gdp_df["Country Code"].isin(selected_countries))
        & (gdp_df["Year"] <= to_year)
        & (from_year <= gdp_df["Year"])
    ]
    fig = px.line(filtered_gdp_df, x="Year", y="GDP", color="Country Code", title="GDP over time")
    chart_html = pyo.plot(fig, output_type="div", include_plotlyjs=True)
    last_year_df = gdp_df[gdp_df["Year"] == to_year]
    first_year_df = gdp_df[gdp_df["Year"] == from_year]
    metrics = []
    for country in selected_countries:
        first_row = first_year_df[first_year_df["Country Code"] == country]
        last_row = last_year_df[last_year_df["Country Code"] == country]
        if not first_row.empty and not last_row.empty:
            first_gdp = first_row["GDP"].values[0] / 1000000000
            last_gdp = last_row["GDP"].values[0] / 1000000000
            if not np.isnan(first_gdp) and first_gdp > 0:
                growth = f"{last_gdp / first_gdp:,.2f}x"
            else:
                growth = "n/a"
            metrics.append({"country": country, "value": f"{last_gdp:,.0f}B", "delta": growth})
        else:
            metrics.append({"country": country, "value": "N/A", "delta": "n/a"})
    return {
        "from_year": from_year,
        "to_year": to_year,
        "selected_countries": selected_countries,
        "countries": countries,
        "min_year": min_year,
        "max_year": max_year,
        "chart_html": chart_html,
        "metrics": metrics,
    }


@app.get("/gdp", response_class=HTMLResponse)
async def gdp_get(request: Request):
    context = _build_gdp_context(None, None, None)
    context["request"] = request
    return templates.TemplateResponse("gdp.html", context)


@app.post("/gdp", response_class=HTMLResponse)
async def gdp_post(
    request: Request,
    from_year: int = Form(...),
    to_year: int = Form(...),
    countries: Optional[List[str]] = Form(None),
):
    context = _build_gdp_context(from_year, to_year, countries or [])
    context["request"] = request
    return templates.TemplateResponse("gdp.html", context)


@app.api_route("/ipl", methods=["GET", "POST"], response_class=HTMLResponse)
@app.api_route("/ipl/", methods=["GET", "POST"], response_class=HTMLResponse)
async def ipl(request: Request, option: Optional[str] = Form(None), t1: Optional[str] = Form(None), t2: Optional[str] = Form(None), team: Optional[str] = Form(None), batsman: Optional[str] = Form(None), bowler: Optional[str] = Form(None)):
    teams = sorted(set(matches["Team1"]).union(set(matches["Team2"])))
    result = {}
    if request.method == "POST" and option:
        if option == "Teams":
            result = teams_api()
            result["type"] = "teams"
        elif option == "Team vs Team" and t1 and t2:
            result = team_v_team_api(t1, t2)
            result.update({"type": "team_vs_team", "t1": t1, "t2": t2})
        elif option == "Team Record" and team:
            result = team_record_api(team)
            result.update({"type": "team_record", "team": team})
        elif option == "Batsman Stats" and batsman:
            data = batsman_api(batsman)
            result = {"type": "batsman", "data": data, "batsman": batsman}
        elif option == "Bowler Stats" and bowler:
            data = bowler_api(bowler)
            result = {"type": "bowler", "data": data, "bowler": bowler}
    return templates.TemplateResponse(
        "ipl.html",
        {"request": request, "result": result, "all_players": all_players, "teams": teams},
    )


@app.get("/weather", response_class=HTMLResponse)
async def weather_get(request: Request):
    return templates.TemplateResponse("weather.html", {"request": request})


@app.post("/weather", response_class=HTMLResponse)
async def weather_post(request: Request, city: str = Form(...)):
    weather_key = os.getenv("WEATHER_API_KEY")
    if not weather_key:
        return templates.TemplateResponse(
            "weather.html", {"request": request, "error": "API key not found."}
        )
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"APPID": weather_key, "q": city, "units": "metric"}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        weather_data = response.json()
        city_name = weather_data["name"]
        conditions = weather_data["weather"][0]["description"].capitalize()
        temp = weather_data["main"]["temp"]
        icon = weather_data["weather"][0]["icon"]
        weather_info = {
            "city": city_name,
            "conditions": conditions,
            "temp": temp,
            "icon": f"/static/weather_icons/{icon}.png",
        }
        return templates.TemplateResponse(
            "weather.html", {"request": request, "weather": weather_info}
        )
    except requests.exceptions.RequestException as exc:
        return templates.TemplateResponse(
            "weather.html", {"request": request, "error": f"Error fetching weather: {exc}"}
        )


@app.get("/skill", response_class=HTMLResponse)
async def skill_get(request: Request):
    return templates.TemplateResponse("skill.html", {"request": request})


@app.post("/skill", response_class=HTMLResponse)
async def skill_post(
    request: Request,
    input_type: str = Form("Paste Resume"),
    resume_text: str = Form(""),
    skills: str = Form(""),
):
    if not ensure_role_data_loaded():
        return templates.TemplateResponse(
            "skill.html",
            {
                "request": request,
                "error": role_model_error
                or "Skill model unavailable. Please try again after downloading the model.",
            },
        )
    if input_type == "Paste Resume":
        user_skills = extract_skills(resume_text)
    elif input_type == "Enter Skills":
        user_skills = [s.strip() for s in skills.split(",") if s.strip()]
    else:
        user_skills = []
    if not user_skills:
        return templates.TemplateResponse(
            "skill.html", {"request": request, "error": "Please provide resume text or skills."}
        )
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
        recommendations.append(
            {
                "title": role["role_title"],
                "description": role["role_description"],
                "score": round(score, 3),
                "missing_skills": missing,
                "plan": plan,
            }
        )
    return templates.TemplateResponse(
        "skill.html",
        {"request": request, "recommendations": recommendations, "user_skills": user_skills},
    )


@app.api_route("/census", methods=["GET", "POST"], response_class=HTMLResponse)
async def census(request: Request, state: Optional[str] = Form(None), state_query: Optional[str] = Query(None)):
    df = pd.read_csv("INDIA CENSUS/india.csv")
    states = sorted(df["State"].unique())
    selected_state = state or state_query or "Overall INDIA"
    if selected_state == "Overall INDIA":
        state_data = df.groupby("State").agg({"Population": "sum", "Latitude": "mean", "Longitude": "mean"}).reset_index()
        import folium

        folium_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
        for _, row in state_data.iterrows():
            folium.Marker(location=[row["Latitude"], row["Longitude"]], popup=f"{row['State']}: {row['Population']:,}").add_to(folium_map)
        map_html = folium_map._repr_html_()
        filtered_df = df.head(20)
    else:
        filtered_df = df[df["State"] == selected_state].head(20)
        import folium

        folium_map = folium.Map(location=[filtered_df["Latitude"].mean(), filtered_df["Longitude"].mean()], zoom_start=7)
        for _, row in filtered_df.iterrows():
            folium.Marker(location=[row["Latitude"], row["Longitude"]], popup=f"{row['District']}: {row['Population']:,}").add_to(folium_map)
        map_html = folium_map._repr_html_()
    return templates.TemplateResponse(
        "census.html",
        {
            "request": request,
            "df": filtered_df.to_html(index=False),
            "states": states,
            "selected_state": selected_state,
            "map_html": map_html,
        },
    )


@app.get("/attendance", response_class=HTMLResponse)
async def attendance(request: Request):
    return templates.TemplateResponse(
        "attendance.html",
        {
            "request": request,
            "message": "Attendance System: Face recognition based.",
        },
    )


@app.get("/house", response_class=HTMLResponse)
async def house_get(request: Request):
    return templates.TemplateResponse("house.html", {"request": request})


@app.post("/house", response_class=HTMLResponse)
async def house_post(
    request: Request,
    lot_area: float = Form(...),
    year_built: int = Form(...),
    first_flr_sf: float = Form(...),
    second_flr_sf: float = Form(...),
    full_bath: int = Form(...),
    bedroom_abv_gr: int = Form(...),
    tot_rms_abv_grd: int = Form(...),
    overall_qual: int = Form(...),
    overall_cond: int = Form(...),
):
    if not house_model:
        return templates.TemplateResponse(
            "house.html", {"request": request, "prediction": "Model is missing on server."}
        )
    input_data = pd.DataFrame(
        {
            "LotArea": [lot_area],
            "YearBuilt": [year_built],
            "1stFlrSF": [first_flr_sf],
            "2ndFlrSF": [second_flr_sf],
            "FullBath": [full_bath],
            "BedroomAbvGr": [bedroom_abv_gr],
            "TotRmsAbvGrd": [tot_rms_abv_grd],
            "OverallQual": [overall_qual],
            "OverallCond": [overall_cond],
        }
    )
    imputer = SimpleImputer(strategy="median")
    input_data = pd.DataFrame(imputer.fit_transform(input_data), columns=input_data.columns)
    prediction = house_model.predict(input_data)[0]
    return templates.TemplateResponse(
        "house.html", {"request": request, "prediction": round(float(prediction), 2)}
    )


@app.get("/sql", response_class=HTMLResponse)
async def sql_get(request: Request):
    return templates.TemplateResponse(
        "sql.html",
        {
            "request": request,
            "message": "Click 'Compare' to run SQL comparison on db1.sql and db2.sql.",
        },
    )


@app.post("/sql", response_class=HTMLResponse)
async def sql_post(request: Request):
    try:
        subprocess.run([sys.executable, "compare_sql.py"], cwd="SQL COMPARISION", check=True)
        summary_df = pd.read_csv("SQL COMPARISION/summary/db_comparison_summary.csv")
        report_df = pd.read_csv("SQL COMPARISION/reports/db_comparison_report.csv")
        return templates.TemplateResponse(
            "sql.html",
            {
                "request": request,
                "summary": summary_df.to_html(),
                "report": report_df.to_html(),
            },
        )
    except subprocess.CalledProcessError as exc:
        return templates.TemplateResponse(
            "sql.html", {"request": request, "error": f"Error running comparison: {exc}"}
        )


@app.get("/faq", response_class=HTMLResponse)
async def faq_get(request: Request):
    return templates.TemplateResponse(
        "faq.html", {"request": request, "message": "Enter a URL to extract FAQs from websites."}
    )


@app.post("/faq", response_class=HTMLResponse)
async def faq_post(request: Request, url: str = Form(...)):
    try:
        faq_module_path = os.path.join(os.getcwd(), "FAQ EXTRACTOR")
        sys.path.insert(0, faq_module_path)
        from app import extract_faqs_from_html, fetch_url
    except Exception as exc:  # pylint: disable=broad-except
        return templates.TemplateResponse(
            "faq.html", {"request": request, "error": f"Error loading FAQ extractor: {exc}"}
        )
    try:
        html = fetch_url(url)
        faqs = extract_faqs_from_html(html)
        return templates.TemplateResponse("faq.html", {"request": request, "faqs": faqs[:10]})
    except Exception as exc:  # pylint: disable=broad-except
        return templates.TemplateResponse(
            "faq.html", {"request": request, "error": f"Error extracting FAQs: {exc}"}
        )
    finally:
        if sys.path[0] == faq_module_path:
            sys.path.pop(0)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
