# Data Vista

Data Vista is a unified collection of data science, machine learning, analytics, and visualization projects in one repository. It combines Flask pages, Streamlit apps, notebooks, and utility scripts into a single portfolio-style platform.

## Highlights

- 10+ integrated projects across ML, NLP, CV, analytics, and dashboards
- One landing application with project-specific pages and templates
- Includes both model training notebooks and runnable app interfaces
- Good reference repository for end-to-end applied data projects

## Included Projects

1. **Diabetes Prediction**
Predicts diabetes risk from health indicators such as glucose, insulin, BMI, and age.

2. **GDP Dashboard**
Interactive GDP trend analysis using historical country-level data and visual charts.

3. **IPL Analytics App**
IPL team, player, and match analysis using ball-by-ball and match datasets.

4. **Skill Advisory**
Resume-to-role recommendation system based on NLP embeddings and similarity search.

5. **India Census Explorer**
State/district level demographic exploration with map and census-based insights.

6. **Weather App**
City weather lookup with condition icons and external API integration.

7. **Student Attendance (Face Recognition)**
OpenCV-based registration, training, and attendance marking pipeline.

8. **Advanced House Price Prediction**
Feature engineering and model workflows for house price estimation.

9. **SQL Comparison Tool**
Compares SQL schemas/content and generates CSV reports and summaries.

10. **FAQ Extractor**
Extracts and serves FAQ-style Q&A from text sources with NLP pipelines.

## Technology Stack

- Backend: Flask, FastAPI (experimental file present), Streamlit
- Data and ML: Pandas, NumPy, scikit-learn, sentence-transformers
- Visualization: Plotly
- Computer Vision: OpenCV, face-recognition
- Parsing/Scraping/Utilities: BeautifulSoup, lxml, requests, sqlparse, python-dotenv

## Prerequisites

- Python 3.8+
- pip

## Setup

1. Clone repository

```bash
git clone https://github.com/Shikher-jain/Data-Vista.git
cd Data-Vista
```

2. Create and activate virtual environment

```bash
python -m venv .venv
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Configure environment variables (Weather App)

Create a `.env` file inside `WEATHER APP`:

```env
WEATHER_API_KEY=your_api_key_here
```

## Run

Start the main Flask app:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

## Repository Layout

```text
Data-Vista/
|-- app.py
|-- main.py
|-- fastapi_app.py
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

- Some projects are standalone and can be run from their own folders.
- Model/data files are included in subdirectories where required.
- For project-specific commands, check each subfolder README.

## Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-change`
3. Commit: `git commit -m "Describe your change"`
4. Push: `git push origin feature/your-change`
5. Open a pull request

## Author

Shikher Jain  
GitHub: https://github.com/Shikher-jain

## Support

- Open a GitHub issue for bugs or feature requests
- Submit a pull request for improvements
