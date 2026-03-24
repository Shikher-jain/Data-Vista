# Data Vista

Data Vista is a multi-project data science platform that brings machine learning, analytics, visualization, NLP, and utility workflows into one application.

The current primary backend entrypoint is FastAPI via main.py.

## App Info

- Primary app entrypoint: main.py
- Recommended run command: uvicorn main:app --reload

## What is Included

- Diabetes prediction
- GDP dashboard and country trend analysis
- IPL team and player analytics
- Skill advisory based on resume and skills
- India census exploration
- Weather lookup with icon support
- Student attendance and student management workflows
- Advanced house price prediction
- SQL schema comparison with summary and report output
- FAQ extraction from website content
- Laptop price prediction module

## Tech Stack

- Backend: FastAPI, Streamlit (primary entrypoint is FastAPI via main.py; some subprojects also use Flask independently)
- Data and ML: pandas, numpy, scikit-learn, sentence-transformers
- Visualization: Plotly
- CV: OpenCV, face-recognition
- Parsing and utilities: requests, sqlparse, BeautifulSoup, lxml, python-dotenv

## Requirements

- Python 3.8+
- pip

## Setup

1. Clone the repository

```bash
git clone https://github.com/Shikher-jain/Data-Vista.git
cd Data-Vista
```

2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Configure environment variables

Create a .env file in the project root (or ensure WEATHER_API_KEY is available in environment):

```env
WEATHER_API_KEY=your_api_key_here
```

## Run (FastAPI)

Development:

```bash
uvicorn main:app --reload
```

Production (matches the included `Procfile`):

```bash
gunicorn main:app -k uvicorn.workers.UvicornWorker
```

Open:

```text
http://127.0.0.1:8000/
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Key Paths

```text
Data-Vista/
|-- main.py
|-- requirements.txt
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

## Notes

- Some subprojects are standalone and can be run independently.
- SQL comparison supports either pasted SQL or uploaded .sql files for db1 and db2.
- GDP dashboard displays full country names while preserving code-based filtering.

## Contributing

1. Fork the repository
2. Create a branch
3. Commit your changes
4. Push the branch
5. Open a pull request

## Author

Shikher Jain

GitHub:
https://github.com/Shikher-jain

## Support

- Open an issue for bugs or feature requests
- Submit pull requests for improvements
