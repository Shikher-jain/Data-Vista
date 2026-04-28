# Data Vista

Data Vista is a multi-project analytics platform with one unified FastAPI backend and multiple AI/ML mini-apps rendered with Jinja templates.

### Live Demo: 
- v1 [https://data-vista-t67v.onrender.com](https://data-vista-t67v.onrender.com)
- v2 [https://data-vista-fastapi.onrender.com](https://data-vista-fastapi.onrender.com)
- v3 [https://data-vista-1-m4y8.onrender.com](https://data-vista-1-m4y8.onrender.com)

### Video Walkthrough: [https://drive.google.com/file/d/1E-wnjWx6sytiW072_Jjs_jnhIZHunCn1/view?usp=sharing](https://drive.google.com/file/d/1E-wnjWx6sytiW072_Jjs_jnhIZHunCn1/view?usp=sharing)

## What Changed (Architecture)

- Canonical backend entrypoint is now `main.py`.
- `app.py` and `fastapi_app.py` are compatibility wrappers that re-export the same FastAPI app.
- Shared data now uses SQLite (`data_vista.db`) through `data_store.py` for better scalability.
- Student records are persisted in SQLite and mirrored to `Student_Management/students.json` for backward compatibility with Streamlit/Tkinter tools.
- GDP dashboard data is seeded into SQLite from `GDP DASHBOARD/data/gdp_data.csv` on startup.
- Routes have stronger validation and friendlier user-facing error responses.

## Project Highlights

- Diabetes Prediction
- GDP Dashboard
- IPL Analytics
- Skill Advisory (embedding based recommendations)
- India Census Explorer
- Weather App
- House Price Prediction
- Laptop Price Prediction
- SQL Comparison Tool
- FAQ Extractor
- Student Management
- Student Attendance (standalone folder workflow)

## Tech Stack

- Backend: FastAPI, Jinja2, Uvicorn/Gunicorn
- Data/ML: pandas, numpy, scikit-learn, sentence-transformers, joblib, pickle
- Visualization: Plotly, Folium
- Storage: SQLite (new shared app storage)
- Utilities: requests, python-dotenv

## Quick Start

1. Clone and enter the repo

```bash
git clone https://github.com/Shikher-jain/Data-Vista.git
cd Data-Vista
```

2. Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Configure environment variables

Create `.env` in the repository root:

```env
WEATHER_API_KEY=your_openweather_api_key
```

## Run Locally

Preferred:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Compatibility launcher:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:8000/
```

## Production Entry

`Procfile` is configured for ASGI:

```text
web: gunicorn -k uvicorn.workers.UvicornWorker main:app
```

## Storage Notes

- Shared database file: `data_vista.db`
- Created and initialized at startup by `DataVistaStore` in `data_store.py`
- Seeds GDP data from CSV on first run
- Migrates student data from JSON if present on first run
- Keeps JSON in sync after student add/delete operations

## Validation and Error Handling

The main app now includes:

- Form numeric parsing helpers with clear validation messages
- URL validation for FAQ extraction
- Better weather API error reporting
- FastAPI exception handlers for request validation and server errors
- Generic fallback error template: `templates/error.html`

## Key Files

- `main.py`: canonical FastAPI app and all routes
- `data_store.py`: SQLite service, migration, and seed logic
- `app.py`: compatibility launcher for older commands
- `fastapi_app.py`: compatibility import for older module references
- `templates/`: UI pages for all modules

## Repo Layout (Top-Level)

```text
Data-Vista/
|-- main.py
|-- app.py
|-- fastapi_app.py
|-- data_store.py
|-- templates/
|-- static/
|-- ADV HOUSE PREDICTION/
|-- DIABETES PREDICTION/
|-- FAQ EXTRACTOR/
|-- GDP DASHBOARD/
|-- INDIA CENSUS/
|-- IPL APP/
|-- SKILL ADVISORY/
|-- SQL COMPARISION/
|-- StudentAttendance/
|-- Student_Management/
|-- WEATHER APP/
`-- laptop-price-predictor-regression-project/
```

## Contribution

1. Fork the repository
2. Create a feature branch
3. Make focused changes with tests where possible
4. Open a pull request with a clear description
