import os
import asyncio
import json
import inspect
from pathlib import Path
import importlib.util
from typing import Any, Dict, List, Optional, Tuple
import ast
import re
import subprocess
import warnings
import sys
import sqlite3
import time
from urllib.parse import urlparse

try:
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:
    InconsistentVersionWarning = None

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

if InconsistentVersionWarning is not None:
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

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
laptop_pipe = None
laptop_df = None
attendance_process = None

BASE_DIR = Path(__file__).resolve().parent

SQL_QUERY_PAIRS: List[Dict[str, str]] = [
    {
        "id": "employee_spend",
        "label": "Employee Spend (Slow vs Optimized)",
        "slow_query": (
            "SELECT e.name, "
            "(SELECT SUM(o.amount) FROM orders o WHERE o.emp_id = e.id) AS total_amount "
            "FROM employees e "
            "ORDER BY total_amount DESC LIMIT 20;"
        ),
        "optimized_query": (
            "SELECT e.name, COALESCE(SUM(o.amount), 0) AS total_amount "
            "FROM employees e "
            "LEFT JOIN orders o ON o.emp_id = e.id "
            "GROUP BY e.id, e.name "
            "ORDER BY total_amount DESC LIMIT 20;"
        ),
    },
    {
        "id": "amount_filter",
        "label": "Order Amount Filter (Slow vs Optimized)",
        "slow_query": (
            "SELECT * FROM orders "
            "WHERE CAST(amount AS TEXT) LIKE '5%';"
        ),
        "optimized_query": (
            "SELECT order_id, emp_id, amount, status, order_date "
            "FROM orders "
            "WHERE amount >= 500 AND amount < 600;"
        ),
    },
    {
        "id": "product_lookup",
        "label": "Product Lookup (Slow vs Optimized)",
        "slow_query": (
            "SELECT * FROM products "
            "WHERE LOWER(product_name) = LOWER('Product 050');"
        ),
        "optimized_query": (
            "SELECT product_id, product_name, price "
            "FROM products "
            "WHERE product_name = 'Product 050';"
        ),
    },
    {
        "id": "monthly_summary",
        "label": "Monthly Revenue (Slow vs Optimized)",
        "slow_query": (
            "SELECT strftime('%Y-%m', order_date) AS month_bucket, "
            "SUM(amount) AS total_amount "
            "FROM orders "
            "GROUP BY strftime('%Y-%m', order_date) "
            "ORDER BY month_bucket;"
        ),
        "optimized_query": (
            "SELECT substr(order_date, 1, 7) AS month_bucket, "
            "SUM(amount) AS total_amount "
            "FROM orders "
            "GROUP BY month_bucket "
            "ORDER BY month_bucket;"
        ),
    },
]

SQL_EXAMPLE_SCHEMA = """
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS employees;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS departments;

CREATE TABLE departments (
    dept_id INTEGER PRIMARY KEY,
    dept_name TEXT NOT NULL
);

CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    dept TEXT NOT NULL,
    email TEXT
);

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    price REAL NOT NULL
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    emp_id INTEGER NOT NULL,
    amount REAL NOT NULL,
    status TEXT NOT NULL,
    order_date TEXT NOT NULL
);

CREATE INDEX idx_orders_emp_id ON orders(emp_id);
CREATE INDEX idx_orders_amount ON orders(amount);
CREATE INDEX idx_orders_status_date ON orders(status, order_date);
CREATE INDEX idx_products_name ON products(product_name);
"""


def parse_checkbox(form_data, field_name: str, default: bool = False) -> bool:
    raw = form_data.get(field_name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def get_sql_pair_by_id(pair_id: Optional[str]) -> Dict[str, str]:
    if not pair_id:
        return SQL_QUERY_PAIRS[0]
    for pair in SQL_QUERY_PAIRS:
        if pair["id"] == pair_id:
            return pair
    return SQL_QUERY_PAIRS[0]


def resolve_groq_api_key() -> str:
    return (os.getenv("GROQ_API_KEY") or "").strip()


def call_groq_chat_completion(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    max_tokens: int = 300,
    temperature: float = 0.2,
) -> Tuple[Optional[str], Optional[str]]:
    if not api_key:
        return None, "Groq API key is required."

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            return None, "Groq response did not contain any summary text."
        return content.strip(), None
    except requests.exceptions.RequestException as exc:
        return None, f"Groq API request failed: {exc}"
    except Exception as exc:
        return None, f"Groq response parsing failed: {exc}"


def build_sql_benchmark_llm_summary(benchmark: Dict[str, Any], api_key: str) -> Tuple[Optional[str], Optional[str]]:
    slow = benchmark.get("slow", {})
    optimized = benchmark.get("optimized", {})
    prompt = (
        "Summarize the SQL benchmark in 5 concise bullet points. Include: "
        "which query is faster, estimated gain, one likely reason, one indexing tip, and one caution.\n\n"
        f"Slow query avg ms: {slow.get('avg_ms')}\n"
        f"Optimized query avg ms: {optimized.get('avg_ms')}\n"
        f"Faster query: {benchmark.get('faster_query')}\n"
        f"Fallback sample data used: {benchmark.get('used_sample_data')}\n"
        f"Slow query text: {slow.get('query')}\n"
        f"Optimized query text: {optimized.get('query')}"
    )
    return call_groq_chat_completion(
        system_prompt="You are a SQL performance analyst. Be precise and practical.",
        user_prompt=prompt,
        api_key=api_key,
        max_tokens=280,
        temperature=0.1,
    )


def build_faq_llm_cleanup(
    faqs: List[Dict[str, str]],
    api_key: str,
    max_items: int = 12,
) -> Tuple[Optional[List[Dict[str, str]]], Optional[str]]:
    if not faqs:
        return [], None

    sample = faqs[:max_items]
    sample_json = json.dumps(sample, ensure_ascii=False)
    prompt = (
        "Clean and normalize the FAQ JSON array. Keep only useful entries. "
        "Return JSON only. Rules: remove duplicates, trim whitespace, remove promotional fluff, "
        "keep clear question and answer fields only.\n\n"
        f"Input JSON:\n{sample_json}"
    )
    content, error = call_groq_chat_completion(
        system_prompt="You clean FAQ datasets and return strict JSON only.",
        user_prompt=prompt,
        api_key=api_key,
        max_tokens=900,
        temperature=0.0,
    )
    if error:
        return None, error
    if not content:
        return None, "Empty Groq cleanup response."

    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).strip()

    try:
        parsed = json.loads(text)
    except Exception as exc:
        return None, f"Could not parse LLM cleanup JSON: {exc}"

    if not isinstance(parsed, list):
        return None, "LLM cleanup response was not a JSON array."

    cleaned: List[Dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if question and answer:
            cleaned.append({"question": question, "answer": answer})

    if not cleaned:
        return None, "LLM cleanup did not return valid FAQ entries."

    return cleaned, None


def _seed_sql_benchmark_sample_data(conn: sqlite3.Connection) -> None:
    conn.executescript(SQL_EXAMPLE_SCHEMA)

    dept_names = ["HR", "IT", "Finance", "Sales", "Marketing"]
    conn.executemany(
        "INSERT INTO departments (dept_id, dept_name) VALUES (?, ?)",
        [(idx + 1, name) for idx, name in enumerate(dept_names)],
    )

    employees = []
    for idx in range(1, 301):
        dept = dept_names[(idx - 1) % len(dept_names)]
        employees.append((idx, f"Employee {idx:03d}", dept, f"employee{idx:03d}@example.com"))
    conn.executemany(
        "INSERT INTO employees (id, name, dept, email) VALUES (?, ?, ?, ?)",
        employees,
    )

    products = []
    for idx in range(1, 201):
        price = round(15 + (idx * 7.35), 2)
        products.append((idx, f"Product {idx:03d}", price))
    conn.executemany(
        "INSERT INTO products (product_id, product_name, price) VALUES (?, ?, ?)",
        products,
    )

    statuses = ["Pending", "Completed", "Cancelled"]
    orders = []
    for idx in range(1, 5001):
        emp_id = (idx % 300) + 1
        amount = round(50 + ((idx * 37) % 950) + ((idx % 7) * 0.19), 2)
        status = statuses[idx % len(statuses)]
        month = (idx % 12) + 1
        day = (idx % 28) + 1
        order_date = f"2024-{month:02d}-{day:02d}"
        orders.append((idx, emp_id, amount, status, order_date))
    conn.executemany(
        "INSERT INTO orders (order_id, emp_id, amount, status, order_date) VALUES (?, ?, ?, ?, ?)",
        orders,
    )
    conn.commit()


def _has_user_tables(conn: sqlite3.Connection) -> bool:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
    )
    return cursor.fetchone() is not None


def _create_sql_benchmark_connection(schema_sql: str) -> Tuple[sqlite3.Connection, bool, Optional[str]]:
    warning = None
    conn = sqlite3.connect(":memory:")
    used_sample_data = False

    if schema_sql.strip():
        try:
            conn.executescript(schema_sql)
        except Exception as exc:
            warning = f"Could not load provided SQL schema into SQLite: {exc}. Used built-in sample dataset instead."
            conn.close()
            conn = sqlite3.connect(":memory:")
            _seed_sql_benchmark_sample_data(conn)
            return conn, True, warning

    if not _has_user_tables(conn):
        _seed_sql_benchmark_sample_data(conn)
        used_sample_data = True
        if warning is None:
            warning = "No tables found in provided SQL. Used built-in sample dataset for benchmark."

    return conn, used_sample_data, warning


def _execute_query_once(conn: sqlite3.Connection, query: str) -> Tuple[int, int]:
    normalized = query.strip().rstrip(";")
    if not normalized:
        raise ValueError("Query is empty.")

    cursor = conn.cursor()
    starts_with = normalized.lstrip().lower()
    is_read_query = starts_with.startswith(("select", "with", "pragma", "explain"))

    if is_read_query:
        cursor.execute(normalized)
        rows = cursor.fetchall()
        return len(rows), max(cursor.rowcount, 0)

    conn.execute("BEGIN")
    try:
        cursor.execute(normalized)
        affected = max(cursor.rowcount, 0)
    finally:
        conn.rollback()
    return 0, affected


def _run_query_benchmark(
    conn: sqlite3.Connection,
    query: str,
    warmup_runs: int,
    measured_runs: int,
) -> Dict[str, Any]:
    normalized = query.strip().rstrip(";")
    if not normalized:
        return {"query": query, "error": "Query is empty."}

    explain_lines: List[str] = []
    try:
        explain_cursor = conn.execute(f"EXPLAIN QUERY PLAN {normalized}")
        explain_rows = explain_cursor.fetchall()
        explain_lines = [" | ".join(str(part) for part in row) for row in explain_rows]
    except Exception as exc:
        explain_lines = [f"Explain not available: {exc}"]

    try:
        for _ in range(max(warmup_runs, 0)):
            _execute_query_once(conn, normalized)

        durations_ms: List[float] = []
        row_count = 0
        affected_rows = 0
        for _ in range(max(measured_runs, 1)):
            start = time.perf_counter()
            row_count, affected_rows = _execute_query_once(conn, normalized)
            elapsed_ms = (time.perf_counter() - start) * 1000
            durations_ms.append(elapsed_ms)

        return {
            "query": normalized,
            "avg_ms": round(float(np.mean(durations_ms)), 3),
            "median_ms": round(float(np.median(durations_ms)), 3),
            "p95_ms": round(float(np.percentile(durations_ms, 95)), 3),
            "min_ms": round(float(np.min(durations_ms)), 3),
            "max_ms": round(float(np.max(durations_ms)), 3),
            "row_count": row_count,
            "affected_rows": affected_rows,
            "runs": len(durations_ms),
            "explain": explain_lines,
        }
    except Exception as exc:
        return {
            "query": normalized,
            "error": str(exc),
            "explain": explain_lines,
        }


def benchmark_query_pair(
    schema_sql: str,
    slow_query: str,
    optimized_query: str,
    warmup_runs: int,
    measured_runs: int,
) -> Dict[str, Any]:
    warning_messages: List[str] = []

    def _run_benchmark_once(schema_text: str) -> Tuple[Dict[str, Any], Dict[str, Any], bool, Optional[str]]:
        conn, used_sample_data, warning = _create_sql_benchmark_connection(schema_text)
        try:
            slow_stats = _run_query_benchmark(conn, slow_query, warmup_runs, measured_runs)
            optimized_stats = _run_query_benchmark(conn, optimized_query, warmup_runs, measured_runs)
        finally:
            conn.close()
        return slow_stats, optimized_stats, used_sample_data, warning

    slow_stats, optimized_stats, used_sample_data, warning = _run_benchmark_once(schema_sql)
    if warning:
        warning_messages.append(warning)

    if (slow_stats.get("error") or optimized_stats.get("error")) and not used_sample_data:
        slow_stats, optimized_stats, retry_used_sample, retry_warning = _run_benchmark_once("")
        used_sample_data = retry_used_sample
        warning_messages.append(
            "Provided schema could not execute selected benchmark queries. Retried on built-in sample dataset."
        )
        if retry_warning:
            warning_messages.append(retry_warning)

    faster_query = "n/a"
    speedup_pct = None
    if not slow_stats.get("error") and not optimized_stats.get("error"):
        slow_avg = float(slow_stats["avg_ms"])
        optimized_avg = float(optimized_stats["avg_ms"])
        if optimized_avg < slow_avg:
            faster_query = "optimized"
            speedup_pct = round(((slow_avg - optimized_avg) / max(slow_avg, 0.0001)) * 100, 2)
        elif slow_avg < optimized_avg:
            faster_query = "slow"
            speedup_pct = round(((optimized_avg - slow_avg) / max(optimized_avg, 0.0001)) * 100, 2)
        else:
            faster_query = "tie"
            speedup_pct = 0.0

    return {
        "slow": slow_stats,
        "optimized": optimized_stats,
        "faster_query": faster_query,
        "speedup_pct": speedup_pct,
        "used_sample_data": used_sample_data,
        "warnings": warning_messages,
    }


def build_sql_page_context(request: Request, **overrides: Any) -> Dict[str, Any]:
    default_pair = SQL_QUERY_PAIRS[0]
    context: Dict[str, Any] = {
        "request": request,
        "message": "Paste SQL for DB1 and DB2, or upload db1.sql and db2.sql, then run comparison.",
        "query_pairs": SQL_QUERY_PAIRS,
        "selected_query_pair": default_pair["id"],
        "db1_sql_input": "",
        "db2_sql_input": "",
        "query_slow_input": default_pair["slow_query"],
        "query_optimized_input": default_pair["optimized_query"],
        "warmup_runs_input": 1,
        "measured_runs_input": 6,
        "use_llm": False,
        "llm_insight": None,
        "llm_error": None,
        "benchmark": None,
        "sample_schema_sql": SQL_EXAMPLE_SCHEMA.strip(),
    }
    context.update(overrides)
    return context


def build_faq_page_context(request: Request, **overrides: Any) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "request": request,
        "message": "Enter a URL to extract FAQs from websites (supports static and many dynamic pages).",
        "url_input": "",
        "fetch_attempts": [],
        "crawl_depth": 1,
        "max_follow_links": 12,
        "max_workers": 8,
        "timeout": 20,
        "min_answer_len": 20,
        "allow_dynamic": True,
        "reuse_cache": True,
        "use_llm_cleanup": False,
        "cache_file": None,
        "cache_hit": False,
        "pages": [],
        "faqs": [],
        "page_count": 0,
        "faq_count": 0,
        "duration": None,
        "warnings": [],
        "llm_error": None,
    }
    context.update(overrides)
    return context


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

    artifacts_dir = BASE_DIR / "DIABETES PREDICTION" / "outputs" / "advanced_pipeline" / "artifacts"
    metadata_path = BASE_DIR / "DIABETES PREDICTION" / "outputs" / "advanced_pipeline" / "run_metadata.json"
    model_path = artifacts_dir / "tuned_random_forest.joblib"
    scaler_path = artifacts_dir / "engineered_feature_scaler.joblib"
    artifact_paths = (model_path, scaler_path, metadata_path)

    if (
        advanced_diabetes_model is not None
        and advanced_diabetes_scaler is not None
        and advanced_diabetes_metadata is not None
    ):
        return advanced_diabetes_model, advanced_diabetes_scaler, advanced_diabetes_metadata, None

    if advanced_diabetes_error is not None and not all(path.exists() for path in artifact_paths):
        return None, None, None, advanced_diabetes_error

    if advanced_diabetes_error is not None:
        advanced_diabetes_error = None

    try:
        advanced_diabetes_model = joblib.load(model_path)
        advanced_diabetes_scaler = joblib.load(scaler_path)
        advanced_diabetes_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        advanced_diabetes_error = None
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


def normalize_smoking_history_value(value: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        return normalized

    legacy_aliases = {
        "no info": "Not applicable",
        "not applicable": "Not applicable",
        "ever": "former",
        "not current": "former",
    }
    return legacy_aliases.get(normalized.lower(), normalized)


def load_house_model():
    global house_model
    if house_model is None:
        try:
            house_model = joblib.load("ADV HOUSE PREDICTION/house_model.pkl")
        except FileNotFoundError:
            house_model = None
    return house_model


def _apply_numpy_pickle_compat() -> None:
    """Create aliases for numpy internals across major-version pickle differences."""
    try:
        import numpy.core as np_core
        import numpy.core.numeric as np_core_numeric

        sys.modules.setdefault("numpy._core", np_core)
        sys.modules.setdefault("numpy._core.numeric", np_core_numeric)
    except Exception:
        # If aliasing fails, we still have a rebuild fallback.
        return


def _rebuild_laptop_artifacts(project_dir: Path) -> Tuple[bool, Optional[str]]:
    rebuild_script = project_dir / "rebuild_model.py"
    if not rebuild_script.exists():
        return False, f"Missing rebuild script: {rebuild_script}"

    try:
        result = subprocess.run(
            [sys.executable, str(rebuild_script)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        details = (result.stdout or "").strip()
        return True, details or None
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or str(exc)).strip()
        return False, details or str(exc)
    except Exception as exc:
        return False, str(exc)


def load_laptop_artifacts() -> Tuple[Any, Optional[pd.DataFrame], Optional[str]]:
    global laptop_pipe, laptop_df

    if laptop_pipe is not None and laptop_df is not None:
        return laptop_pipe, laptop_df, None

    project_dir = BASE_DIR / "laptop-price-predictor-regression-project"
    model_path = project_dir / "pipe.pkl"
    df_path = project_dir / "df.pkl"

    if not model_path.exists() or not df_path.exists():
        return None, None, (
            "Laptop model files are missing. Expected pipe.pkl and df.pkl in "
            "laptop-price-predictor-regression-project."
        )

    def _load() -> Tuple[Any, pd.DataFrame]:
        with open(model_path, "rb") as model_file:
            pipe = pickle.load(model_file)
        with open(df_path, "rb") as df_file:
            df = pickle.load(df_file)
        return pipe, df

    load_error: Exception = RuntimeError("Unknown laptop artifact load failure")

    try:
        laptop_pipe, laptop_df = _load()
        return laptop_pipe, laptop_df, None
    except ModuleNotFoundError as exc:
        load_error = exc
        if "numpy._core.numeric" in str(exc):
            _apply_numpy_pickle_compat()
            try:
                laptop_pipe, laptop_df = _load()
                return laptop_pipe, laptop_df, None
            except Exception as retry_exc:
                load_error = retry_exc
    except Exception as exc:
        load_error = exc

    rebuilt, rebuild_details = _rebuild_laptop_artifacts(project_dir)
    if rebuilt:
        try:
            laptop_pipe, laptop_df = _load()
            return laptop_pipe, laptop_df, None
        except Exception as exc:
            return None, None, f"Laptop model rebuild succeeded but loading still failed: {exc}"

    error_message = f"Model or data file load failed: {load_error}"
    if rebuild_details:
        error_message = f"{error_message}. Rebuild attempt failed: {rebuild_details}"

    return None, None, error_message


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


def _filter_supported_kwargs(func: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return kwargs

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs

    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def load_student_records():
    try:
        records = DATA_STORE.list_students()
        items = []
        for row in records:
            name = row["name"]
            grade = float(row["grade"])
            range_upper = float(row.get("range_upper") or 10.0)
            if range_upper <= 0:
                range_upper = 10.0
            items.append(
                {
                    "name": name,
                    "grade": grade,
                    "range_upper": range_upper,
                    "marks": f"{grade:g}/{range_upper:g}",
                    "decimal_result": f"{grade / range_upper:g}",
                }
            )
        items.sort(key=lambda x: x["name"].lower())
        return {row["name"]: row["grade"] for row in records}, items, None
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

def resolve_grade_range(form_data) -> Tuple[float, float, str]:
    selected_range = (form_data.get("grade_range") or "1-10").strip()
    presets = {
        "1-5": (1.0, 5.0),
        "1-10": (1.0, 10.0),
        "1-100": (1.0, 100.0),
    }

    if selected_range in presets:
        lower, upper = presets[selected_range]
        return lower, upper, selected_range

    if selected_range.lower() == "custom":
        upper = parse_float(form_data, "custom_max", "Custom maximum grade", min_value=1)
        lower = 1.0
        if upper < lower:
            raise ValueError("Custom maximum grade must be at least 1.")
        return lower, upper, f"1-{upper:g}"

    raise ValueError("Please choose a valid grade range option.")


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


@app.get("/parameters", response_class=HTMLResponse)
def parameters_reference(request: Request):
    return templates.TemplateResponse("parameters.html", {"request": request})


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
                smoking_history = normalize_smoking_history_value(form.get("smoking_history_adv") or "")
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
    states = sorted(df["State"].dropna().unique())

    for col in ["Population", "Latitude", "Longitude", "literacy_rate", "sex_ratio", "Male", "Female", "Literate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if request.method == "POST":
        form = await request.form()
        selected_state = (form.get("state") or "Overall INDIA").strip()
        selected_x_metric = (form.get("x_metric") or "literacy_rate").strip()
        selected_y_metric = (form.get("y_metric") or "sex_ratio").strip()
    else:
        selected_state = "Overall INDIA"
        selected_x_metric = "literacy_rate"
        selected_y_metric = "sex_ratio"

    if selected_state != "Overall INDIA" and selected_state not in states:
        selected_state = "Overall INDIA"

    is_overall_view = selected_state == "Overall INDIA"

    if is_overall_view:
        chart_df = (
            df.groupby("State", as_index=False)
            .agg(
                Population=("Population", "sum"),
                Latitude=("Latitude", "mean"),
                Longitude=("Longitude", "mean"),
                literacy_rate=("literacy_rate", "mean"),
                sex_ratio=("sex_ratio", "mean"),
                Districts=("District", "nunique"),
            )
        )
        label_col = "State"
        count_label = "States"
        table_columns = ["State", "Districts", "Population", "literacy_rate", "sex_ratio"]
        bar_limit = 15
    else:
        chart_df = df[df["State"] == selected_state].copy()
        label_col = "District"
        count_label = "Districts"
        table_columns = ["District", "Population", "Male", "Female", "Literate", "literacy_rate", "sex_ratio"]
        bar_limit = 20

    metric_candidates = [
        col
        for col in chart_df.columns
        if col != label_col and pd.api.types.is_numeric_dtype(chart_df[col]) and chart_df[col].notna().any()
    ]
    preferred_metric_order = ["literacy_rate", "sex_ratio", "Population", "Male", "Female", "Literate", "Districts", "Latitude", "Longitude"]
    ordered_metrics = [col for col in preferred_metric_order if col in metric_candidates] + [col for col in metric_candidates if col not in preferred_metric_order]

    if not ordered_metrics:
        ordered_metrics = ["Population"]

    if selected_x_metric not in ordered_metrics:
        selected_x_metric = "literacy_rate" if "literacy_rate" in ordered_metrics else ordered_metrics[0]

    if selected_y_metric not in ordered_metrics:
        selected_y_metric = "sex_ratio" if "sex_ratio" in ordered_metrics else ordered_metrics[0]

    if selected_y_metric == selected_x_metric and len(ordered_metrics) > 1:
        for metric in ordered_metrics:
            if metric != selected_x_metric:
                selected_y_metric = metric
                break

    metric_options = [{"value": metric, "label": metric.replace("_", " ").title()} for metric in ordered_metrics]

    map_source = chart_df.dropna(subset=["Latitude", "Longitude", "Population"]).copy()
    population_source = chart_df.dropna(subset=["Population"]).copy()
    literacy_source = chart_df.dropna(subset=["literacy_rate", "sex_ratio", "Population"]).copy()
    scatter_source = chart_df.dropna(subset=[selected_x_metric, selected_y_metric, "Population"]).copy()

    map_plot_html = None
    if not map_source.empty:
        map_color_col = "literacy_rate" if map_source["literacy_rate"].notna().any() else "Population"
        map_hover_columns = {
            "Population": ":,.0f",
            "literacy_rate": ":.2f",
            "sex_ratio": ":.2f",
        }
        if is_overall_view and "Districts" in map_source.columns:
            map_hover_columns["Districts"] = ":,.0f"

        map_fig = px.scatter_geo(
            map_source,
            lat="Latitude",
            lon="Longitude",
            size="Population",
            color=map_color_col,
            hover_name=label_col,
            hover_data=map_hover_columns,
            projection="natural earth",
            title=("India Census Interactive Map (State Level)" if is_overall_view else f"{selected_state} Census Interactive Map (District Level)"),
            height=520,
        )
        map_fig.update_geos(
            lataxis_range=[6, 38],
            lonaxis_range=[68, 98],
            showcountries=True,
            countrycolor="#7b8ca0",
            showland=True,
            landcolor="#f2f6f4",
        )
        map_fig.update_layout(margin=dict(l=0, r=0, t=55, b=0), coloraxis_colorbar_title="Literacy")
        map_plot_html = pyo.plot(
            map_fig,
            output_type="div",
            include_plotlyjs=False,
            config={"displaylogo": False, "responsive": True},
        )

    population_bar_html = None
    if not population_source.empty:
        top_population = population_source.sort_values("Population", ascending=False).head(bar_limit)
        bar_fig = px.bar(
            top_population,
            x=label_col,
            y="Population",
            color="literacy_rate" if top_population["literacy_rate"].notna().any() else "Population",
            hover_data={"Population": ":,.0f", "literacy_rate": ":.2f", "sex_ratio": ":.2f"},
            title=("Top States by Population" if is_overall_view else f"Top Districts by Population in {selected_state}"),
            height=420,
        )
        bar_fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), xaxis_title=label_col, yaxis_title="Population")
        population_bar_html = pyo.plot(
            bar_fig,
            output_type="div",
            include_plotlyjs=False,
            config={"displaylogo": False, "responsive": True},
        )

    demographics_scatter_html = None
    if not scatter_source.empty:
        scatter_hover_data = {"Population": ":,.0f"}
        for metric in [selected_x_metric, selected_y_metric, "literacy_rate", "sex_ratio"]:
            if metric in scatter_source.columns and metric not in scatter_hover_data:
                series = scatter_source[metric].dropna()
                if not series.empty and np.all(np.isclose(series, np.round(series))):
                    scatter_hover_data[metric] = ":,.0f"
                else:
                    scatter_hover_data[metric] = ":.2f"

        x_axis_label = selected_x_metric.replace("_", " ").title()
        y_axis_label = selected_y_metric.replace("_", " ").title()
        scatter_fig = px.scatter(
            scatter_source,
            x=selected_x_metric,
            y=selected_y_metric,
            size="Population",
            color="Population",
            hover_name=label_col,
            hover_data=scatter_hover_data,
            title=(f"{x_axis_label} vs {y_axis_label}" if is_overall_view else f"{x_axis_label} vs {y_axis_label} in {selected_state}"),
            height=420,
        )
        scatter_fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), xaxis_title=x_axis_label, yaxis_title=y_axis_label)
        demographics_scatter_html = pyo.plot(
            scatter_fig,
            output_type="div",
            include_plotlyjs=False,
            config={"displaylogo": False, "responsive": True},
        )

    total_population = int(population_source["Population"].sum()) if not population_source.empty else 0
    average_literacy = float(literacy_source["literacy_rate"].mean()) if not literacy_source.empty else None
    average_sex_ratio = float(literacy_source["sex_ratio"].mean()) if not literacy_source.empty else None
    unit_count = int(chart_df[label_col].nunique()) if not chart_df.empty else 0

    top_unit = None
    if not population_source.empty:
        top_row = population_source.sort_values("Population", ascending=False).iloc[0]
        top_unit = f"{top_row[label_col]} ({int(top_row['Population']):,})"

    display_df = chart_df.copy()
    if not display_df.empty:
        display_df = display_df.sort_values("Population", ascending=False).head(30)
        display_df = display_df[[col for col in table_columns if col in display_df.columns]].copy()

        for int_col in ["Population", "Male", "Female", "Literate", "Districts"]:
            if int_col in display_df.columns:
                display_df[int_col] = display_df[int_col].apply(lambda x: "-" if pd.isna(x) else f"{int(round(float(x))):,}")

        for float_col in ["literacy_rate", "sex_ratio"]:
            if float_col in display_df.columns:
                display_df[float_col] = display_df[float_col].apply(lambda x: "-" if pd.isna(x) else f"{float(x):.2f}")

        table_html = display_df.to_html(index=False, classes="table table-striped table-hover")
    else:
        table_html = None

    return templates.TemplateResponse(
        "census.html",
        {
            "request": request,
            "states": states,
            "selected_state": selected_state,
            "selected_x_metric": selected_x_metric,
            "selected_y_metric": selected_y_metric,
            "metric_options": metric_options,
            "selected_scope": "Overall INDIA" if is_overall_view else selected_state,
            "total_population": f"{total_population:,}",
            "average_literacy": None if average_literacy is None else f"{average_literacy:.2f}",
            "average_sex_ratio": None if average_sex_ratio is None else f"{average_sex_ratio:.2f}",
            "unit_count": unit_count,
            "count_label": count_label,
            "top_unit": top_unit,
            "map_plot_html": map_plot_html,
            "population_bar_html": population_bar_html,
            "demographics_scatter_html": demographics_scatter_html,
            "census_table": table_html,
        },
    )


@app.api_route("/attendance", methods=["GET", "POST"], response_class=HTMLResponse)
async def attendance(request: Request):
    attendance_dir = BASE_DIR / "StudentAttendance"
    register_script = attendance_dir / "register.py"
    train_script = attendance_dir / "train.py"
    mark_script = attendance_dir / "mark_attendance.py"

    def launch_attendance_gui(script_path: Path, *script_args: str) -> str:
        global attendance_process

        if not script_path.exists():
            raise FileNotFoundError(f"Missing attendance script: {script_path}")

        if attendance_process is not None and attendance_process.poll() is None:
            return "Attendance GUI is already running."

        attendance_process = subprocess.Popen(
            [sys.executable, str(script_path), *script_args],
            cwd=str(attendance_dir),
        )
        return f"{script_path.stem.replace('_', ' ').title()} launched. The local window should now be open on this machine."

    async def run_training_script() -> str:
        if not train_script.exists():
            raise FileNotFoundError(f"Missing attendance script: {train_script}")

        completed = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, str(train_script)],
            cwd=str(attendance_dir),
            capture_output=True,
            text=True,
        )
        output = (completed.stdout or "").strip()
        error_output = (completed.stderr or "").strip()
        if completed.returncode != 0:
            raise RuntimeError(error_output or output or "Training failed without a message.")
        return output or "Training completed successfully."

    launch_message = None
    if request.method == "POST":
        form = await request.form()
        action = str(form.get("action") or "mark").strip().lower()
        student_name = str(form.get("student_name") or "").strip()

        try:
            if action == "register":
                if not student_name:
                    raise ValueError("Enter a student name before starting registration.")
                launch_message = launch_attendance_gui(register_script, student_name)
            elif action == "train":
                launch_message = await run_training_script()
            elif action == "mark":
                launch_message = launch_attendance_gui(mark_script)
            else:
                raise ValueError("Unknown attendance action.")
        except Exception as exc:
            launch_message = f"Could not complete the requested attendance action: {exc}"

    msg = (
        "Use the buttons below to run StudentAttendance/register.py, StudentAttendance/train.py, or StudentAttendance/mark_attendance.py. "
        "Register first, then train, then mark attendance."
    )
    return templates.TemplateResponse("attendance.html", {"request": request, "message": msg, "launch_message": launch_message})


@app.api_route("/students", methods=["GET", "POST"], response_class=HTMLResponse)
async def student_management(request: Request):
    _, records, error = load_student_records()
    message = None
    grade_range = "1-10"
    custom_max = ""
    name_input = ""
    grade_input = ""

    if request.method == "POST":
        form = await request.form()
        action = (form.get("action") or "").strip().lower()
        grade_range = (form.get("grade_range") or "1-10").strip()
        custom_max = (form.get("custom_max") or "").strip()
        name_input = (form.get("name") or "").strip()
        grade_input = (form.get("grade") or "").strip()
        name = name_input

        if action == "add":
            if not name or not grade_input:
                message = "Name and grade are required."
            else:
                try:
                    min_grade, max_grade, range_label = resolve_grade_range(form)
                    grade = parse_float(form, "grade", "Grade", min_value=min_grade, max_value=max_grade)
                    DATA_STORE.upsert_student(name, grade, max_grade)
                    message = f"Saved {name}. Grade range used: {range_label}."
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
            "grade_range": grade_range,
            "custom_max": custom_max,
            "name_input": name_input,
            "grade_input": grade_input,
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

    pipe, df, load_error = load_laptop_artifacts()
    if load_error is not None or pipe is None or df is None:
        context["error"] = load_error or "Laptop model artifacts are unavailable."
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
            feature_columns = [column for column in df.columns if column != "Price"]
            query = pd.DataFrame(
                [[company, type_name, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, osys]],
                columns=feature_columns,
            )
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
    query_pair_id: Optional[str] = Form(None),
    warmup_runs_raw: Optional[str] = Form(None),
    measured_runs_raw: Optional[str] = Form(None),
    use_llm: Optional[str] = Form(None),
):
    def _looks_like_query(text: str) -> bool:
        candidate = (text or "").strip().lower()
        if not candidate:
            return False
        token_match = re.match(r"^([a-z_]+)", candidate)
        token = token_match.group(1) if token_match else ""
        return token in {"select", "with", "explain", "insert", "update", "delete"}

    db1_sql_input = (db1_sql or "").strip()
    db2_sql_input = (db2_sql or "").strip()
    selected_pair = get_sql_pair_by_id(query_pair_id)

    if _looks_like_query(db1_sql_input):
        query_slow_input = db1_sql_input
    else:
        query_slow_input = selected_pair["slow_query"]

    if _looks_like_query(db2_sql_input):
        query_optimized_input = db2_sql_input
    else:
        query_optimized_input = selected_pair["optimized_query"]

    try:
        warmup_runs = int(str(warmup_runs_raw).strip()) if warmup_runs_raw is not None else 1
    except ValueError:
        warmup_runs = 1
    warmup_runs = max(0, min(warmup_runs, 10))

    try:
        measured_runs = int(str(measured_runs_raw).strip()) if measured_runs_raw is not None else 6
    except ValueError:
        measured_runs = 6
    measured_runs = max(1, min(measured_runs, 30))

    should_use_llm = str(use_llm or "").strip().lower() in {"1", "true", "yes", "on"}

    if request.method == "POST":
        sql_dir = Path("SQL COMPARISION")
        db1_path = sql_dir / "db1.sql"
        db2_path = sql_dir / "db2.sql"
        db1_text = ""
        db2_text = ""

        has_file_input = bool(db1_file and db1_file.filename) or bool(db2_file and db2_file.filename)
        has_text_input = bool(db1_sql_input) or bool(db2_sql_input)

        base_context = build_sql_page_context(
            request,
            selected_query_pair=selected_pair["id"],
            db1_sql_input=db1_sql_input,
            db2_sql_input=db2_sql_input,
            query_slow_input=query_slow_input,
            query_optimized_input=query_optimized_input,
            warmup_runs_input=warmup_runs,
            measured_runs_input=measured_runs,
            use_llm=should_use_llm,
        )

        if has_file_input:
            if not (db1_file and db1_file.filename and db2_file and db2_file.filename):
                return templates.TemplateResponse(
                    "sql.html",
                    {**base_context, "error": "Please upload both files: db1.sql and db2.sql."},
                )

            if not db1_file.filename.lower().endswith(".sql") or not db2_file.filename.lower().endswith(".sql"):
                return templates.TemplateResponse(
                    "sql.html",
                    {**base_context, "error": "Only .sql files are allowed for upload."},
                )

            db1_bytes = await db1_file.read()
            db2_bytes = await db2_file.read()
            if not db1_bytes.strip() or not db2_bytes.strip():
                return templates.TemplateResponse(
                    "sql.html",
                    {**base_context, "error": "Uploaded SQL files must not be empty."},
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
            if not (db1_sql_input and db2_sql_input):
                return templates.TemplateResponse(
                    "sql.html",
                    {**base_context, "error": "Please provide SQL content for both DB1 and DB2."},
                )

            db1_text = db1_sql_input
            db2_text = db2_sql_input

            textareas_query_mode = _looks_like_query(db1_text) and _looks_like_query(db2_text)
            if not textareas_query_mode:
                db1_path.write_text(db1_text, encoding="utf-8")
                db2_path.write_text(db2_text, encoding="utf-8")

        else:
            return templates.TemplateResponse(
                "sql.html",
                {**base_context, "error": "Paste SQL in both text areas or upload both .sql files."},
            )

        comparison_message = None
        comparison_error = None
        summary_html = None
        report_html = None

        textareas_query_mode = _looks_like_query(db1_text) and _looks_like_query(db2_text)
        if textareas_query_mode and not has_file_input:
            comparison_message = (
                "Using DB1/DB2 textareas as query inputs. "
                "Schema comparison step was skipped for this run."
            )
        else:
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
                comparison_message = "Comparison completed successfully."
                if process.stdout:
                    comparison_message = f"{comparison_message} {process.stdout.strip()}"
                summary_html = summary_df.to_html(index=False, classes="table table-striped table-sm")
                report_html = report_df.to_html(index=False, classes="table table-striped table-sm")
            except subprocess.CalledProcessError as exc:
                details = exc.stderr.strip() if exc.stderr else str(exc)
                comparison_error = f"Error running comparison: {details}"

        benchmark = benchmark_query_pair(
            schema_sql=db1_text,
            slow_query=query_slow_input,
            optimized_query=query_optimized_input,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
        )

        llm_insight = None
        llm_error = None
        if should_use_llm:
            api_key = resolve_groq_api_key()
            if not api_key:
                llm_error = "Groq API key is required when LLM summary is enabled. Set GROQ_API_KEY in server environment variables."
            else:
                llm_insight, llm_error = build_sql_benchmark_llm_summary(benchmark, api_key)

        message = comparison_message or "Benchmark report generated."
        if benchmark.get("warnings"):
            message = f"{message} {benchmark['warnings'][0]}"

        return templates.TemplateResponse(
            "sql.html",
            build_sql_page_context(
                request,
                message=message,
                error=comparison_error,
                summary=summary_html,
                report=report_html,
                benchmark=benchmark,
                llm_insight=llm_insight,
                llm_error=llm_error,
                selected_query_pair=selected_pair["id"],
                db1_sql_input=db1_text,
                db2_sql_input=db2_text,
                query_slow_input=query_slow_input,
                query_optimized_input=query_optimized_input,
                warmup_runs_input=warmup_runs,
                measured_runs_input=measured_runs,
                use_llm=should_use_llm,
            ),
        )

    return templates.TemplateResponse("sql.html", build_sql_page_context(request))


@app.api_route("/faq", methods=["GET", "POST"], response_class=HTMLResponse)
async def faq(request: Request):
    if request.method == "POST":
        form = await request.form()
        url = (form.get("url") or "").strip()

        def _safe_int(name: str, default: int, min_value: int, max_value: int, label: str) -> Tuple[int, Optional[str]]:
            raw = (form.get(name) or "").strip()
            if raw == "":
                return default, None
            try:
                value = int(raw)
            except ValueError:
                return default, f"{label} must be a whole number."
            if value < min_value or value > max_value:
                return default, f"{label} must be between {min_value} and {max_value}."
            return value, None

        crawl_depth, depth_error = _safe_int("crawl_depth", 1, 0, 5, "Crawler depth")
        max_follow_links, links_error = _safe_int("max_follow_links", 12, 1, 120, "Max linked pages")
        max_workers, workers_error = _safe_int("max_workers", 8, 1, 20, "Max workers")
        timeout, timeout_error = _safe_int("timeout", 20, 5, 90, "Timeout")
        min_answer_len, answer_error = _safe_int("min_answer_len", 20, 0, 3000, "Minimum answer length")

        allow_dynamic = parse_checkbox(form, "allow_dynamic", default=True)
        reuse_cache = parse_checkbox(form, "reuse_cache", default=True)
        use_llm_cleanup = parse_checkbox(form, "use_llm_cleanup", default=False)

        base_context = build_faq_page_context(
            request,
            url_input=url,
            crawl_depth=crawl_depth,
            max_follow_links=max_follow_links,
            max_workers=max_workers,
            timeout=timeout,
            min_answer_len=min_answer_len,
            allow_dynamic=allow_dynamic,
            reuse_cache=reuse_cache,
            use_llm_cleanup=use_llm_cleanup,
        )

        number_error = depth_error or links_error or workers_error or timeout_error or answer_error
        if number_error:
            return templates.TemplateResponse("faq.html", {**base_context, "error": number_error})

        if not url:
            return templates.TemplateResponse("faq.html", {**base_context, "error": "Please provide a URL."})
        if not is_valid_http_url(url):
            return templates.TemplateResponse(
                "faq.html",
                {**base_context, "error": "Please provide a valid URL starting with http:// or https://."},
            )

        try:
            extractor, err = load_faq_extractor()
            if err:
                return templates.TemplateResponse("faq.html", {**base_context, "error": err})

            if not hasattr(extractor, "run_extraction"):
                return templates.TemplateResponse(
                    "faq.html",
                    {**base_context, "error": "Loaded FAQ extractor module does not expose run_extraction()."},
                )

            result = extractor.run_extraction(
                url,
                crawl_depth,
                max_workers,
                timeout=timeout,
                min_answer_len=min_answer_len,
                reuse_cache=reuse_cache,
                max_pages=max_follow_links,
                allow_dynamic=allow_dynamic,
            )

            faqs = [
                faq
                for faq in result.get("faqs", [])
                if len(str(faq.get("answer", "")).strip()) >= min_answer_len
            ]
            pages = [{"url": page_url, "faqs": "n/a"} for page_url in result.get("urls", [])]
            cache_file = str(result.get("qna_file")) if result.get("qna_file") else None
            cache_hit = bool(result.get("cached"))
            fetch_attempts = result.get("fetch_attempts", [])
            warning_msgs = [msg for msg in result.get("warnings", []) if str(msg).strip()]

            llm_error = None
            if use_llm_cleanup and faqs:
                api_key = resolve_groq_api_key()
                if not api_key:
                    llm_error = "Groq API key is required when LLM cleanup is enabled. Set GROQ_API_KEY in server environment variables."
                else:
                    sample_size = min(12, len(faqs))
                    cleaned_sample, llm_error = build_faq_llm_cleanup(faqs[:sample_size], api_key, max_items=sample_size)
                    if cleaned_sample:
                        faqs = cleaned_sample + faqs[sample_size:]

            if not faqs:
                error_text = "No FAQ pairs were detected for the provided URL."
                if warning_msgs:
                    error_text = f"{error_text} Details: {' | '.join(warning_msgs[:2])}"

                return templates.TemplateResponse(
                    "faq.html",
                    {
                        **base_context,
                        "error": error_text,
                        "pages": pages,
                        "faqs": [],
                        "fetch_attempts": fetch_attempts,
                        "llm_error": llm_error,
                        "cache_file": cache_file,
                        "cache_hit": cache_hit,
                        "site_name": result.get("site_name"),
                        "qna_name": Path(result["qna_file"]).name if result.get("qna_file") else None,
                        "page_count": result.get("page_count", len(pages)),
                        "faq_count": 0,
                        "duration": result.get("duration"),
                        "warnings": warning_msgs,
                    },
                )

            success_message = (
                f"Loaded {len(faqs)} FAQs from cached JSONL for this URL."
                if cache_hit
                else f"Extracted {len(faqs)} FAQs from the target site."
            )

            return templates.TemplateResponse(
                "faq.html",
                {
                    **base_context,
                    "message": success_message,
                    "faqs": faqs[:30],
                    "pages": pages,
                    "fetch_attempts": fetch_attempts,
                    "llm_error": llm_error,
                    "cache_file": cache_file,
                    "cache_hit": cache_hit,
                    "site_name": result.get("site_name"),
                    "qna_name": Path(result["qna_file"]).name if result.get("qna_file") else None,
                    "page_count": result.get("page_count", len(pages)),
                    "faq_count": len(faqs),
                    "duration": result.get("duration"),
                    "warnings": warning_msgs,
                },
            )
        except Exception as exc:
            return templates.TemplateResponse(
                "faq.html",
                {**base_context, "error": f"Error extracting FAQs: {exc}"},
            )

    return templates.TemplateResponse("faq.html", build_faq_page_context(request))


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