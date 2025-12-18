# Data Vista 📊

A comprehensive data science and machine learning web application that integrates 10+ AI/ML projects into a unified platform. Built with Flask, this application showcases various data analysis, prediction, and visualization capabilities across multiple domains.

## 🌟 Features

Data Vista is a collection of integrated projects, each serving a unique purpose:

### 1. **Diabetes Prediction** 🏥
Predict the likelihood of diabetes based on medical parameters like glucose level, insulin, age, and BMI using machine learning models.

### 2. **GDP Dashboard** 📈
Interactive dashboard to visualize and analyze GDP data of countries from 1960 to 2022 with dynamic charts powered by Plotly.

### 3. **IPL App** 🏏
Comprehensive IPL (Indian Premier League) statistics and analysis including:
- Team vs Team performance
- Player statistics and comparisons
- Match insights and historical data

### 4. **Skill Advisory** 💼
Get personalized career and skill recommendations based on your resume text using NLP embeddings and nearest neighbor matching.

### 5. **India Census** 🇮🇳
Analyze and visualize Indian census data with interactive insights and demographic information.

### 6. **Weather App** ☁️
Real-time weather information for cities worldwide with detailed forecasts and weather icons.

### 7. **Attendance System** 👤
Face recognition-based attendance marking system using OpenCV and face recognition technology.

### 8. **Advanced House Price Prediction** 🏠
Predict house prices using advanced machine learning models with feature engineering and selection.

### 9. **SQL Comparison Tool** 🔍
Compare and analyze SQL database structures and schemas with detailed difference reports.

### 10. **FAQ Extractor** ❓
Extract frequently asked questions from text or documents using sentence transformers and NLP.

## 🛠️ Technology Stack

- **Backend Framework**: Flask
- **Data Analysis**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, Sentence Transformers
- **Visualization**: Plotly, Streamlit
- **Computer Vision**: OpenCV, face-recognition
- **Web Scraping**: BeautifulSoup, lxml
- **Others**: Requests, python-dotenv, SQLparse

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/Shikher-jain/Data-Vista.git
cd Data-Vista
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
   
   For the Weather App, create a `.env` file in the `WEATHER APP` directory:
```bash
WEATHER_API_KEY=your_api_key_here
```

## 💻 Usage

1. **Run the Flask application**
```bash
python app.py
```

2. **Access the application**

Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

3. **Navigate through projects**

Use the navigation bar to explore different projects or click on project cards on the homepage.

## 📁 Project Structure

```
Data-Vista/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── static/                     # Static assets (CSS, JS, images)
│   ├── style.css
│   ├── script.js
│   └── weather_icons/
├── templates/                  # HTML templates
│   ├── base.html
│   ├── index.html
│   └── [project-specific templates]
├── ADV HOUSE PREDICTION/       # House price prediction models
├── ATTENDANCE SYSTEM/          # Face recognition attendance
├── DIABETES PREDICTION/        # Diabetes prediction ML model
├── FAQ EXTRACTOR/              # FAQ extraction tool
├── GDP DASHBOARD/              # GDP data visualization
├── INDIA CENSUS/               # Census data analysis
├── IPL APP/                    # IPL statistics and analysis
├── SKILL ADVISORY/             # Career recommendation system
├── SQL COMPARISON/             # SQL comparison utility
└── WEATHER APP/                # Weather information service
```

## 🎯 Individual Project Details

Each subdirectory contains its own README with specific details:

- **ADV HOUSE PREDICTION**: Advanced house price prediction using machine learning
- **ATTENDANCE SYSTEM**: Real-time face recognition for attendance tracking
- **DIABETES PREDICTION**: Medical diagnosis prediction system
- **FAQ EXTRACTOR**: Automated FAQ generation from text
- **GDP DASHBOARD**: Economic data visualization tool
- **INDIA CENSUS**: Demographic analysis platform
- **IPL APP**: Cricket statistics and analytics
- **SKILL ADVISORY**: AI-powered career guidance
- **SQL COMPARISON**: Database schema comparison tool
- **WEATHER APP**: Weather forecast application

## 🎨 Features Highlights

- **Responsive Design**: Modern, mobile-friendly UI with dark/light theme toggle
- **Interactive Visualizations**: Dynamic charts and graphs using Plotly
- **Machine Learning Models**: Pre-trained models for various prediction tasks
- **Real-time Data**: Live weather updates and data processing
- **Unified Platform**: All projects accessible from a single interface

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see individual project directories for specific license information.

## 👨‍💻 Author

**Shikher Jain**

- GitHub: [@Shikher-jain](https://github.com/Shikher-jain)

## 🙏 Acknowledgments

- IPL official statistics for cricket data
- Streamlit for interactive dashboards
- Flask community for the excellent web framework
- All open-source libraries and their maintainers

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Submit a pull request with improvements

---

⭐ If you find this project useful, please consider giving it a star!
