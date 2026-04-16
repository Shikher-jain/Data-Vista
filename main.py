import os
import json
from pathlib import Path
import importlib.util
from typing import Any, Dict, Optional, Tuple
import ast
import re
import subprocess
import warnings
import sys
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pandas as pd
import numpy as np
import pickle
import joblib
import requests
import plotly.express as px
import plotly.offline as pyo
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# Suppress noisy pandas warning seen in IPL stats code paths.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*Series.__getitem__ treating keys as positions.*",
)

load_dotenv()

app = FastAPI(title="Data Vista FastAPI")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Cached artifacts
role_model: Optional[SentenceTransformer] = None
roles_df: Optional[pd.DataFrame] = None
role_embeddings: Optional[np.ndarray] = None
nn: Optional[NearestNeighbors] = None
role_model_error: Optional[str] = None

diabetes_model = None
diabetes_scaler = None
advanced_diabetes_model = None
advanced_diabetes_scaler = None
advanced_diabetes_metadata: Optional[Dict[str, Any]] = None
advanced_diabetes_error: Optional[str] = None
house_model = None
faq_extractor_module = None

BASE_DIR = Path(__file__).resolve().parent


def load_data_store_class():
    module_path = BASE_DIR / "data_store.py"
    spec = importlib.util.spec_from_file_location("data_store_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load data_store module at {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module.DataVistaStore


DataVistaStore = load_data_store_class()

STUDENT_MGMT_DIR = BASE_DIR / "Student_Management"
STUDENT_MGMT_DATA = STUDENT_MGMT_DIR / "students.json"
DATA_STORE = DataVistaStore(
    db_path=BASE_DIR / "data_vista.db",
    students_json_path=STUDENT_MGMT_DATA,
    gdp_csv_path=BASE_DIR / "GDP DASHBOARD" / "data" / "gdp_data.csv",
)
DATA_STORE_ERROR: Optional[str] = None

matches = None
balls = None
all_players = None
batter_data = None
bowler_data = None


def ensure_role_data_loaded() -> bool:
    """Lazily load role data and embeddings."""
    global roles_df, role_model, role_embeddings, nn, role_model_error

    is_serverless = os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME")

    if roles_df is None:
        try:
            roles_df = pd.read_csv("SKILL ADVISORY/roles_catalog_large.csv", quotechar='"', on_bad_lines="skip")
            roles_df.fillna("", inplace=True)
        except Exception as exc:
            role_model_error = f"Could not load roles catalog: {exc}"
            return False

    if role_model is None and role_model_error is None:
        if is_serverless:
            role_model_error = "Sentence transformers disabled in serverless environment to conserve memory."
            return False
        try:
            role_model = SentenceTransformer("all-MiniLM-L6-v2", tokenizer_kwargs={"clean_up_tokenization_spaces": True})
        except Exception as exc:
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
    global diabetes_model, diabetes_scaler
    if diabetes_model is None or diabetes_scaler is None:
        try:
            with open("DIABETES PREDICTION/flask/model.pkl", "rb") as f:
                diabetes_model = pickle.load(f)
            dataset = pd.read_csv("DIABETES PREDICTION/diabetes.csv")
            diabetes_scaler = MinMaxScaler(feature_range=(0, 1))
            diabetes_scaler.fit(dataset.iloc[:, [1, 2, 5, 7]].values)
        except FileNotFoundError:
            diabetes_model = None
            diabetes_scaler = None
    return diabetes_model, diabetes_scaler


def load_advanced_diabetes_artifacts() -> Tuple[Any, Any, Optional[Dict[str, Any]], Optional[str]]:
    global advanced_diabetes_model, advanced_diabetes_scaler, advanced_diabetes_metadata, advanced_diabetes_error

    if (
        advanced_diabetes_model is not None
        and advanced_diabetes_scaler is not None
        and advanced_diabetes_metadata is not None
    ):
        return advanced_diabetes_model, advanced_diabetes_scaler, advanced_diabetes_metadata, None

    if advanced_diabetes_error is not None:
        return None, None, None, advanced_diabetes_error

    artifacts_dir = Path("DIABETES PREDICTION") / "outputs" / "advanced_pipeline" / "artifacts"
    metadata_path = Path("DIABETES PREDICTION") / "outputs" / "advanced_pipeline" / "run_metadata.json"

    try:
        advanced_diabetes_model = joblib.load(artifacts_dir / "tuned_random_forest.joblib")
        advanced_diabetes_scaler = joblib.load(artifacts_dir / "engineered_feature_scaler.joblib")
        advanced_diabetes_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return advanced_diabetes_model, advanced_diabetes_scaler, advanced_diabetes_metadata, None
    except FileNotFoundError:
        advanced_diabetes_model = None
        advanced_diabetes_scaler = None
        advanced_diabetes_metadata = None
        advanced_diabetes_error = (
            "Advanced model artifacts are missing. "
            "Run DIABETES PREDICTION/advanced_diabetes_pipeline.py first."
        )
        return None, None, None, advanced_diabetes_error
    except Exception as exc:
        advanced_diabetes_model = None
        advanced_diabetes_scaler = None
        advanced_diabetes_metadata = None
        advanced_diabetes_error = f"Could not load advanced diabetes artifacts: {exc}"
        return None, None, None, advanced_diabetes_error


def load_house_model():
    global house_model
    if house_model is None:
        try:
            house_model = joblib.load("ADV HOUSE PREDICTION/house_model.pkl")
        except FileNotFoundError:
            house_model = None
    return house_model


def load_faq_extractor():
    global faq_extractor_module
    if faq_extractor_module is not None:
        return faq_extractor_module, None

    module_path = Path(__file__).resolve().parent / "FAQ EXTRACTOR" / "app.py"
    if not module_path.exists():
        return None, f"FAQ extractor module missing at {module_path}"

    spec = importlib.util.spec_from_file_location("faq_extractor_module", module_path)
    if spec is None or spec.loader is None:
        return None, "Could not load FAQ extractor module specification"

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        faq_extractor_module = module
        return faq_extractor_module, None
    except Exception as exc:
        return None, f"Could not import FAQ extractor: {exc}"


def load_student_records():
    try:
        records = DATA_STORE.list_students()
        data = {row["name"]: row["grade"] for row in records}
        items = sorted(({"name": k, "grade": v} for k, v in data.items()), key=lambda x: x["name"].lower())
        return data, items, None
    except Exception as exc:
        return {}, [], f"Could not read student records: {exc}"


def get_gdp_data():
    try:
        return DATA_STORE.load_gdp_data()
    except Exception:
        # Fallback keeps the dashboard operational if the database is unavailable.
        raw_gdp_df = pd.read_csv("GDP DASHBOARD/data/gdp_data.csv")
        min_year = 1960
        max_year = 2022
        gdp_df = raw_gdp_df.melt(
            ["Country Code", "Country Name"],
            [str(x) for x in range(min_year, max_year + 1)],
            "Year",
            "GDP",
        )
        gdp_df["Year"] = pd.to_numeric(gdp_df["Year"])
        return gdp_df


def parse_float(form_data, field_name: str, label: str, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    raw = form_data.get(field_name)
    if raw is None or str(raw).strip() == "":
        raise ValueError(f"{label} is required.")

    try:
        value = float(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid number.") from exc

    if min_value is not None and value < min_value:
        raise ValueError(f"{label} must be at least {min_value}.")
    if max_value is not None and value > max_value:
        raise ValueError(f"{label} must be at most {max_value}.")
    return value


def parse_int(form_data, field_name: str, label: str, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    raw = form_data.get(field_name)
    if raw is None or str(raw).strip() == "":
        raise ValueError(f"{label} is required.")

    try:
        value = int(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be a whole number.") from exc

    if min_value is not None and value < min_value:
        raise ValueError(f"{label} must be at least {min_value}.")
    if max_value is not None and value > max_value:
        raise ValueError(f"{label} must be at most {max_value}.")
    return value


def is_valid_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=422, content={"error": "Invalid request payload.", "details": exc.errors()})
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "title": "Invalid Input",
            "error": "Some fields were missing or invalid. Please review your input and try again.",
        },
        status_code=422,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=500, content={"error": "Internal server error."})
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "title": "Unexpected Error",
            "error": "Something went wrong while processing your request. Please try again.",
        },
        status_code=500,
    )


def load_ipl_data():
    """Lazily load IPL data only when required."""
    global matches, balls, all_players, batter_data, bowler_data

    if matches is None or balls is None:
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

        bowler_data = batter_data.copy()

        def bowler_run(x):
            if x[0] in ["penalty", "legbyes", "byes"]:
                return 0
            return x[1]

        bowler_data["bowler_run"] = bowler_data[["extra_type", "total_run"]].apply(bowler_run, axis=1)

        def bowler_wicket(x):
            if x[0] in ["caught", "caught and bowled", "bowled", "stumped", "lbw", "hit wicket"]:
                return x[1]
            return 0

        bowler_data["isBowlerWicket"] = bowler_data[["kind", "isWicketDelivery"]].apply(bowler_wicket, axis=1)

    return matches, balls, all_players, batter_data, bowler_data


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
    load_ipl_data()
    teams = sorted(set(matches["Team1"]).union(set(matches["Team2"])))
    return {"teams": teams}


def team_v_team_api(t1, t2):
    load_ipl_data()
    valid_teams = sorted(set(matches["Team1"]).union(set(matches["Team2"])))
    if (t1 in valid_teams) and (t2 in valid_teams):
        temp_df = matches[((matches["Team1"] == t1) & (matches["Team2"] == t2)) | ((matches["Team1"] == t2) & (matches["Team2"] == t1))]
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


def all_round(team):
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


def team_record_api(team):
    self_record = all_round(team)
    teams = sorted(set(matches["Team1"]).union(set(matches["Team2"])))
    against = {team2: team_v_team_api(team, team2) for team2 in teams if team2 != team}
    return {team: {"Overall": convert(self_record), "Against": convert(against)}}


def batsman_record(name, df):
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
        highest_score = f"{highest_score}*" if not is_out else str(highest_score)
    else:
        highest_score = np.nan

    not_out = inngs - out
    mom = df[df.Player_of_Match == name].drop_duplicates("ID", keep="first").shape[0]
    return {
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


def batsman_vs_team(batsman, team, df):
    df = df[df.BowlingTeam == team].copy()
    return batsman_record(batsman, df)


def batsman_api(name, balls_df=None):
    load_ipl_data()
    if balls_df is None:
        balls_df = batter_data
    df = balls_df[balls_df.innings.isin([1, 2])]
    self_record = batsman_record(name, df=df)
    teams = sorted(set(matches["Team1"]).union(set(matches["Team2"])))
    against = {team: batsman_vs_team(name, team, df) for team in teams}
    return {name: {"all": convert(self_record), "Against": convert(against)}}


def bowler_record(bowler, df):
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
    best_wicket = gb.sort_values(["isBowlerWicket", "bowler_run"], ascending=[False, True])[["isBowlerWicket", "bowler_run"]].head(1).values
    best_figure = f"{best_wicket[0][0]}/{best_wicket[0][1]}" if best_wicket.size > 0 else np.nan
    mom = df[df.Player_of_Match == bowler].drop_duplicates("ID", keep="first").shape[0]
    return {
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


def bowler_vs_team(bowler, team, df):
    df = df[df.BattingTeam == team].copy()
    return bowler_record(bowler, df)


def bowler_api(bowler, balls_df=None):
    load_ipl_data()
    if balls_df is None:
        balls_df = bowler_data
    df = balls_df[balls_df.innings.isin([1, 2])]
    self_record = bowler_record(bowler, df=df)
    teams = sorted(set(matches["Team1"]).union(set(matches["Team2"])))
    against = {team: bowler_vs_team(bowler, team, df) for team in teams}
    return {bowler: {"all": convert(self_record), "Against": convert(against)}}


def extract_skills(text, vocab=None):
    if vocab is None:
        vocab = [
            "python", "java", "c++", "react", "node", "django", "flask", "sql",
            "tensorflow", "pytorch", "nlp", "cloud", "aws", "docker", "kubernetes",
            "git", "html", "css", "javascript", "linux", "azure", "pandas", "numpy",
        ]
    text_low = text.lower()
    found = []
    for skill in vocab:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text_low):
            found.append(skill)
    return found


def generate_learning_plan(role_title, missing_skills):
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


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "storage_warning": DATA_STORE_ERROR})


@app.get("/favicon.ico")
def favicon():
    icon_path = Path("static/favicon.ico")
    if icon_path.exists():
        return FileResponse(icon_path)
    return {"message": "favicon not found"}


@app.api_route("/diabetes", methods=["GET", "POST"], response_class=HTMLResponse)
async def diabetes(request: Request):
    default_context = {
        "request": request,
        "selected_model": "basic",
        "form_data": {},
        "prediction_text": None,
    }

    if request.method == "POST":
        form = await request.form()
        model_type = (form.get("model_type") or "basic").strip().lower()
        if model_type not in {"basic", "advanced"}:
            model_type = "basic"

        context = {
            "request": request,
            "selected_model": model_type,
            "form_data": {k: str(v) for k, v in form.items()},
            "prediction_text": None,
        }

        if model_type == "advanced":
            model, sc, metadata, load_error = load_advanced_diabetes_artifacts()
            if load_error is not None or model is None or sc is None or metadata is None:
                context["prediction_text"] = load_error or "Advanced model artifacts are missing on the server."
                return templates.TemplateResponse("diabetes.html", context)

            try:
                gender = (form.get("gender_adv") or "").strip()
                smoking_history = (form.get("smoking_history_adv") or "").strip()
                age = parse_float(form, "age_adv", "Age", min_value=0, max_value=120)
                hypertension = parse_int(form, "hypertension_adv", "Hypertension", min_value=0, max_value=1)
                heart_disease = parse_int(form, "heart_disease_adv", "Heart Disease", min_value=0, max_value=1)
                bmi = parse_float(form, "bmi_adv", "BMI", min_value=0)
                hba1c_level = parse_float(form, "hba1c_level_adv", "HbA1c Level", min_value=0)
                blood_glucose_level = parse_float(form, "blood_glucose_level_adv", "Blood Glucose Level", min_value=0)

                encoder_maps = metadata.get("encoder_maps", {}) if isinstance(metadata, dict) else {}
                gender_map = encoder_maps.get("gender", {}) if isinstance(encoder_maps, dict) else {}
                smoking_map = encoder_maps.get("smoking_history", {}) if isinstance(encoder_maps, dict) else {}

                if gender not in gender_map:
                    supported = ", ".join(sorted(gender_map.keys())) or "Female, Male, Other"
                    raise ValueError(f"Unsupported gender value. Use one of: {supported}")
                if smoking_history not in smoking_map:
                    supported = ", ".join(sorted(smoking_map.keys())) or "No Info, current, ever, former, never, not current"
                    raise ValueError(f"Unsupported smoking history value. Use one of: {supported}")

                row = pd.DataFrame(
                    [
                        {
                            "gender": int(gender_map[gender]),
                            "hypertension": hypertension,
                            "heart_disease": heart_disease,
                            "smoking_history": int(smoking_map[smoking_history]),
                            "bmi_age": bmi * age,
                            "glucose_hba1c": blood_glucose_level * hba1c_level,
                            "hypertension_heart": hypertension + heart_disease,
                        }
                    ]
                )

                row[["bmi_age", "glucose_hba1c"]] = sc.transform(row[["bmi_age", "glucose_hba1c"]])
                selected_features = metadata.get("selected_features", []) if isinstance(metadata, dict) else []
                if selected_features:
                    missing_features = [col for col in selected_features if col not in row.columns]
                    if missing_features:
                        raise ValueError(
                            "Advanced model metadata is inconsistent. Missing features: " + ", ".join(missing_features)
                        )
                    row = row[selected_features]
            except ValueError as exc:
                context["prediction_text"] = str(exc)
                return templates.TemplateResponse("diabetes.html", context)

            prediction = model.predict(row)
            pred = "You have Diabetes, please consult a Doctor." if int(prediction[0]) == 1 else "You don't have Diabetes."

            if hasattr(model, "predict_proba"):
                try:
                    risk = float(model.predict_proba(row)[0][1])
                    pred = f"{pred} (Predicted risk score: {risk * 100:.1f}%)"
                except Exception:
                    pass

            context["prediction_text"] = pred
            return templates.TemplateResponse("diabetes.html", context)

        model, sc = load_diabetes_artifacts()
        if model is None or sc is None:
            context["prediction_text"] = "Model artifacts are missing on the server."
            return templates.TemplateResponse("diabetes.html", context)

        try:
            glucose = parse_float(form, "glucose", "Glucose", min_value=0)
            bloodpressure = parse_float(form, "bloodpressure", "Blood Pressure", min_value=0)
            insulin = parse_float(form, "insulin", "Insulin", min_value=0)
            bmi = parse_float(form, "bmi", "BMI", min_value=0)
        except ValueError as exc:
            context["prediction_text"] = str(exc)
            return templates.TemplateResponse("diabetes.html", context)

        final_features = np.array([[glucose, bloodpressure, insulin, bmi]])
        prediction = model.predict(sc.transform(final_features))

        context["prediction_text"] = "You have Diabetes, please consult a Doctor." if int(prediction[0]) == 1 else "You don't have Diabetes."
        return templates.TemplateResponse("diabetes.html", context)

    return templates.TemplateResponse("diabetes.html", default_context)


@app.api_route("/gdp", methods=["GET", "POST"], response_class=HTMLResponse)
async def gdp(request: Request):
    gdp_df = get_gdp_data()
    min_year = int(gdp_df["Year"].min())
    max_year = int(gdp_df["Year"].max())
    country_df = gdp_df[["Country Code", "Country Name"]].drop_duplicates().sort_values("Country Name")
    country_labels = dict(zip(country_df["Country Code"], country_df["Country Name"]))
    valid_codes = set(country_df["Country Code"].tolist())
    countries = [{"code": row["Country Code"], "name": row["Country Name"]} for _, row in country_df.iterrows()]
    error = None

    if request.method == "POST":
        form = await request.form()
        try:
            from_year = parse_int(form, "from_year", "From Year", min_value=min_year, max_value=max_year)
            to_year = parse_int(form, "to_year", "To Year", min_value=min_year, max_value=max_year)
        except ValueError as exc:
            error = str(exc)
            from_year = min_year
            to_year = max_year

        if from_year > to_year:
            error = "From Year cannot be greater than To Year."
            from_year = min_year
            to_year = max_year

        selected_countries = [code for code in form.getlist("countries") if code in valid_codes]
    else:
        from_year = min_year
        to_year = max_year
        selected_countries = ["DEU", "FRA", "GBR", "BRA", "MEX", "JPN"]

    if not selected_countries:
        selected_countries = ["DEU", "FRA", "GBR", "BRA", "MEX", "JPN"]

    filtered_gdp_df = gdp_df[
        (gdp_df["Country Code"].isin(selected_countries))
        & (gdp_df["Year"] <= to_year)
        & (from_year <= gdp_df["Year"])
    ]

    if filtered_gdp_df.empty:
        error = "No GDP data is available for the selected filters."

    fig = px.line(filtered_gdp_df, x="Year", y="GDP", color="Country Name", title="GDP over time")
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
                delta_color = "normal"
            else:
                growth = "n/a"
                delta_color = "off"
            metrics.append({"country": country_labels.get(country, country), "value": f"{last_gdp:,.0f}B", "delta": growth, "delta_color": delta_color})
        else:
            metrics.append({"country": country_labels.get(country, country), "value": "N/A", "delta": "n/a", "delta_color": "off"})

    return templates.TemplateResponse(
        "gdp.html",
        {
            "request": request,
            "chart_html": chart_html,
            "metrics": metrics,
            "from_year": from_year,
            "to_year": to_year,
            "selected_countries": selected_countries,
            "countries": countries,
            "min_year": min_year,
            "max_year": max_year,
            "error": error,
        },
    )


@app.api_route("/ipl", methods=["GET", "POST"], response_class=HTMLResponse)
@app.api_route("/ipl/", methods=["GET", "POST"], response_class=HTMLResponse)
async def ipl(request: Request):
    load_ipl_data()
    teams = sorted(set(matches["Team1"]).union(set(matches["Team2"])))
    error = None

    if request.method == "POST":
        form = await request.form()
        option = form.get("option")
        result = {}

        if option == "Teams":
            result = teams_api()
            result["type"] = "teams"
        elif option == "Team vs Team":
            t1 = form.get("t1")
            t2 = form.get("t2")
            if not t1 or not t2:
                error = "Please select both teams."
            elif t1 == t2:
                error = "Please select two different teams."
            else:
                result = team_v_team_api(t1, t2)
            result["type"] = "team_vs_team"
            result["t1"] = t1
            result["t2"] = t2
        elif option == "Team Record":
            team = form.get("team")
            result = team_record_api(team)
            result["type"] = "team_record"
            result["team"] = team
        elif option == "Batsman Stats":
            batsman = form.get("batsman")
            data = batsman_api(batsman)
            result = {"type": "batsman", "data": data, "batsman": batsman}
        elif option == "Bowler Stats":
            bowler = form.get("bowler")
            data = bowler_api(bowler)
            result = {"type": "bowler", "data": data, "bowler": bowler}
        else:
            error = "Please choose a valid analysis option."

        return templates.TemplateResponse("ipl.html", {"request": request, "result": result, "all_players": all_players, "teams": teams, "error": error})

    return templates.TemplateResponse("ipl.html", {"request": request, "all_players": all_players, "teams": teams, "error": error})


@app.api_route("/weather", methods=["GET", "POST"], response_class=HTMLResponse)
async def weather(request: Request):
    if request.method == "POST":
        form = await request.form()
        city = (form.get("city") or "").strip()
        if not city:
            return templates.TemplateResponse("weather.html", {"request": request, "error": "Please enter a city name."})

        weather_key = os.getenv("WEATHER_API_KEY")
        if not weather_key:
            return templates.TemplateResponse("weather.html", {"request": request, "error": "API key not found."})

        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"APPID": weather_key, "q": city, "units": "metric"}
        try:
            response = requests.get(url, params=params, timeout=10)
            weather_data = response.json()

            if response.status_code != 200:
                message = weather_data.get("message", "Could not fetch weather details.")
                return templates.TemplateResponse("weather.html", {"request": request, "error": f"Weather service error: {message}"})

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
            return templates.TemplateResponse("weather.html", {"request": request, "weather": weather_info})
        except (requests.exceptions.RequestException, KeyError, TypeError) as exc:
            return templates.TemplateResponse("weather.html", {"request": request, "error": f"Error fetching weather: {exc}"})

    return templates.TemplateResponse("weather.html", {"request": request})


@app.api_route("/skill", methods=["GET", "POST"], response_class=HTMLResponse)
async def skill(request: Request):
    if request.method == "POST":
        if not ensure_role_data_loaded():
            return templates.TemplateResponse("skill.html", {"request": request, "error": role_model_error or "Skill advisory model is unavailable right now."})

        form = await request.form()
        input_type = form.get("input_type")
        if input_type == "Paste Resume":
            resume_text = form.get("resume_text", "")
            user_skills = extract_skills(resume_text)
        elif input_type == "Enter Skills":
            skills_text = form.get("skills", "")
            user_skills = [s.strip() for s in skills_text.split(",") if s.strip()]
        else:
            user_skills = []

        if not user_skills:
            return templates.TemplateResponse("skill.html", {"request": request, "error": "Please provide resume text or skills."})

        if role_model is None or nn is None or role_embeddings is None or roles_df is None:
            return templates.TemplateResponse("skill.html", {"request": request, "error": "Skill advisory model is not ready yet."})

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

        return templates.TemplateResponse("skill.html", {"request": request, "recommendations": recommendations, "user_skills": user_skills})

    return templates.TemplateResponse("skill.html", {"request": request})


@app.api_route("/census", methods=["GET", "POST"], response_class=HTMLResponse)
async def census(request: Request):
    df = pd.read_csv("INDIA CENSUS/india.csv")
    states = sorted(df["State"].unique())

    if request.method == "POST":
        form = await request.form()
        selected_state = form.get("state", "Overall INDIA")
    else:
        selected_state = "Overall INDIA"

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
def attendance(request: Request):
    msg = (
        "Use the StudentAttendance module (register, train, then mark_attendance) from the StudentAttendance folder. "
        "This page provides the Streamlit link and local-camera preview only."
    )
    return templates.TemplateResponse("attendance.html", {"request": request, "message": msg})


@app.api_route("/students", methods=["GET", "POST"], response_class=HTMLResponse)
async def student_management(request: Request):
    _, records, error = load_student_records()
    message = None

    if request.method == "POST":
        form = await request.form()
        action = (form.get("action") or "").strip().lower()
        name = (form.get("name") or "").strip()

        if action == "add":
            if not name:
                message = "Name and grade are required."
            else:
                try:
                    grade = parse_float(form, "grade", "Grade")
                    DATA_STORE.upsert_student(name, grade)
                    message = f"Saved {name}."
                except ValueError as exc:
                    message = str(exc)
                except Exception as exc:
                    error = f"Could not save student record: {exc}"
        elif action == "delete":
            if not name:
                message = "Name is required to delete a student."
            else:
                try:
                    deleted = DATA_STORE.delete_student(name)
                    message = f"Deleted {name}." if deleted else f"{name} not found."
                except Exception as exc:
                    error = f"Could not delete student record: {exc}"
        else:
            message = "Unsupported action."

        _, records, refresh_error = load_student_records()
        if refresh_error and not error:
            error = refresh_error

    return templates.TemplateResponse(
        "student_management.html",
        {
            "request": request,
            "records": records,
            "error": error,
            "message": message,
            "data_file": str(DATA_STORE.db_path),
            "streamlit_cmd": "streamlit run Student_Management/stud_managementSTREAMLIT.py",
            "tkinter_cmd": "python Student_Management/stud_managementTK.py",
        },
    )


@app.api_route("/house", methods=["GET", "POST"], response_class=HTMLResponse)
async def house(request: Request):
    if request.method == "POST":
        form = await request.form()
        try:
            lot_area = parse_float(form, "lot_area", "Lot Area", min_value=0)
            year_built = parse_int(form, "year_built", "Year Built", min_value=1800, max_value=2100)
            first_flr_sf = parse_float(form, "first_flr_sf", "1st Floor SF", min_value=0)
            second_flr_sf = parse_float(form, "second_flr_sf", "2nd Floor SF", min_value=0)
            full_bath = parse_int(form, "full_bath", "Full Bathrooms", min_value=0, max_value=20)
            bedroom_abv_gr = parse_int(form, "bedroom_abv_gr", "Bedrooms Above Ground", min_value=0, max_value=20)
            tot_rms_abv_grd = parse_int(form, "tot_rms_abv_grd", "Total Rooms Above Ground", min_value=0, max_value=30)
            overall_qual = parse_int(form, "overall_qual", "Overall Quality", min_value=1, max_value=10)
            overall_cond = parse_int(form, "overall_cond", "Overall Condition", min_value=1, max_value=10)
        except ValueError as exc:
            return templates.TemplateResponse("house.html", {"request": request, "error": str(exc)})

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

        model = load_house_model()
        if model is None:
            return templates.TemplateResponse("house.html", {"request": request, "error": "Model file is missing on the server."})

        prediction = model.predict(input_data)[0]
        return templates.TemplateResponse("house.html", {"request": request, "prediction": round(prediction, 2)})

    return templates.TemplateResponse("house.html", {"request": request})


@app.api_route("/laptop", methods=["GET", "POST"], response_class=HTMLResponse)
async def laptop(request: Request):
    context = {
        "request": request,
        "prediction": None,
        "error": None,
        "companies": [],
        "types": [],
        "cpus": [],
        "gpus": [],
        "oses": [],
        "model_ready": False,
    }

    model_path = os.path.join("laptop-price-predictor-regression-project", "pipe.pkl")
    df_path = os.path.join("laptop-price-predictor-regression-project", "df.pkl")

    try:
        with open(model_path, "rb") as model_file:
            pipe = pickle.load(model_file)
        with open(df_path, "rb") as df_file:
            df = pickle.load(df_file)
    except Exception as exc:
        context["error"] = f"Model or data file missing: {exc}"
        return templates.TemplateResponse("laptop.html", context)

    context.update(
        {
            "companies": sorted(df["Company"].dropna().unique().tolist()),
            "types": sorted(df["TypeName"].dropna().unique().tolist()),
            "cpus": sorted(df["Cpu brand"].dropna().unique().tolist()),
            "gpus": sorted(df["Gpu brand"].dropna().unique().tolist()),
            "oses": sorted(df["os"].dropna().unique().tolist()),
            "model_ready": True,
        }
    )

    if request.method == "POST":
        form = await request.form()
        try:
            company = form["company"]
            type_name = form["type"]
            ram = parse_int(form, "ram", "RAM", min_value=1)
            weight = parse_float(form, "weight", "Weight", min_value=0.1)
            touchscreen = 1 if form["touchscreen"] == "Yes" else 0
            ips = 1 if form["ips"] == "Yes" else 0
            screen_size = parse_float(form, "screen_size", "Screen Size", min_value=1)
            resolution = form["resolution"]
            cpu = form["cpu"]
            hdd = parse_int(form, "hdd", "HDD", min_value=0)
            ssd = parse_int(form, "ssd", "SSD", min_value=0)
            gpu = form["gpu"]
            osys = form["os"]

            if "x" not in resolution:
                raise ValueError("Resolution format must be WIDTHxHEIGHT.")

            x_res = int(resolution.split("x")[0])
            y_res = int(resolution.split("x")[1])
            ppi = ((x_res**2) + (y_res**2)) ** 0.5 / screen_size
            query = np.array([company, type_name, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, osys]).reshape(1, 12)
            pred = int(np.exp(pipe.predict(query)[0]))
            context["prediction"] = f"The predicted price of this configuration is INR {pred}"
        except Exception as exc:
            context["error"] = f"Prediction error: {exc}"

    return templates.TemplateResponse("laptop.html", context)


@app.api_route("/sql", methods=["GET", "POST"], response_class=HTMLResponse)
async def sql(
    request: Request,
    db1_sql: Optional[str] = Form(None),
    db2_sql: Optional[str] = Form(None),
    db1_file: Optional[UploadFile] = File(None),
    db2_file: Optional[UploadFile] = File(None),
):
    if request.method == "POST":
        sql_dir = Path("SQL COMPARISION")
        db1_path = sql_dir / "db1.sql"
        db2_path = sql_dir / "db2.sql"

        has_file_input = bool(db1_file and db1_file.filename) or bool(db2_file and db2_file.filename)
        has_text_input = bool((db1_sql or "").strip()) or bool((db2_sql or "").strip())

        if has_file_input:
            if not (db1_file and db1_file.filename and db2_file and db2_file.filename):
                return templates.TemplateResponse(
                    "sql.html",
                    {
                        "request": request,
                        "error": "Please upload both files: db1.sql and db2.sql.",
                    },
                )

            if not db1_file.filename.lower().endswith(".sql") or not db2_file.filename.lower().endswith(".sql"):
                return templates.TemplateResponse(
                    "sql.html",
                    {
                        "request": request,
                        "error": "Only .sql files are allowed for upload.",
                    },
                )

            db1_bytes = await db1_file.read()
            db2_bytes = await db2_file.read()
            if not db1_bytes.strip() or not db2_bytes.strip():
                return templates.TemplateResponse(
                    "sql.html",
                    {
                        "request": request,
                        "error": "Uploaded SQL files must not be empty.",
                    },
                )

            try:
                db1_text = db1_bytes.decode("utf-8")
            except UnicodeDecodeError:
                db1_text = db1_bytes.decode("latin-1")

            try:
                db2_text = db2_bytes.decode("utf-8")
            except UnicodeDecodeError:
                db2_text = db2_bytes.decode("latin-1")

            db1_path.write_text(db1_text, encoding="utf-8")
            db2_path.write_text(db2_text, encoding="utf-8")

        elif has_text_input:
            if not (db1_sql and db1_sql.strip() and db2_sql and db2_sql.strip()):
                return templates.TemplateResponse(
                    "sql.html",
                    {
                        "request": request,
                        "error": "Please provide SQL content for both DB1 and DB2.",
                    },
                )

            db1_path.write_text(db1_sql.strip(), encoding="utf-8")
            db2_path.write_text(db2_sql.strip(), encoding="utf-8")

        else:
            return templates.TemplateResponse(
                "sql.html",
                {
                    "request": request,
                    "error": "Paste SQL in both text areas or upload both .sql files.",
                },
            )

        try:
            process = subprocess.run(
                [sys.executable, "compare_sql.py"],
                cwd="SQL COMPARISION",
                check=True,
                capture_output=True,
                text=True,
            )
            summary_df = pd.read_csv("SQL COMPARISION/summary/db_comparison_summary.csv")
            report_df = pd.read_csv("SQL COMPARISION/reports/db_comparison_report.csv")
            return templates.TemplateResponse(
                "sql.html",
                {
                    "request": request,
                    "message": "Comparison completed successfully." if not process.stdout else f"Comparison completed successfully. {process.stdout.strip()}",
                    "summary": summary_df.to_html(),
                    "report": report_df.to_html(),
                },
            )
        except subprocess.CalledProcessError as exc:
            details = exc.stderr.strip() if exc.stderr else str(exc)
            return templates.TemplateResponse("sql.html", {"request": request, "error": f"Error running comparison: {details}"})

    return templates.TemplateResponse(
        "sql.html",
        {
            "request": request,
            "message": "Paste SQL for DB1 and DB2, or upload db1.sql and db2.sql, then run comparison.",
        },
    )


@app.api_route("/faq", methods=["GET", "POST"], response_class=HTMLResponse)
async def faq(request: Request):
    if request.method == "POST":
        form = await request.form()
        url = (form.get("url") or "").strip()
        if not url:
            return templates.TemplateResponse("faq.html", {"request": request, "error": "Please provide a URL."})
        if not is_valid_http_url(url):
            return templates.TemplateResponse("faq.html", {"request": request, "error": "Please provide a valid URL starting with http:// or https://."})

        try:
            extractor, err = load_faq_extractor()
            if err:
                return templates.TemplateResponse("faq.html", {"request": request, "error": err})

            html = extractor.fetch_url(url)
            faqs = extractor.extract_faqs_from_html(html)
            if not faqs:
                return templates.TemplateResponse("faq.html", {"request": request, "error": "No FAQ pairs were detected for the provided URL."})
            return templates.TemplateResponse("faq.html", {"request": request, "faqs": faqs[:10]})
        except Exception as exc:
            return templates.TemplateResponse("faq.html", {"request": request, "error": f"Error extracting FAQs: {exc}"})

    return templates.TemplateResponse("faq.html", {"request": request, "message": "Enter a URL to extract FAQs from websites."})


@app.get("/api/teams")
def api_teams():
    return teams_api()


@app.on_event("startup")
def startup_event():
    global DATA_STORE_ERROR
    try:
        DATA_STORE.initialize()
        DATA_STORE_ERROR = None
        print("FastAPI app started successfully with database initialization.")
    except Exception as exc:
        DATA_STORE_ERROR = str(exc)
        print(f"FastAPI app started, but database initialization failed: {exc}")