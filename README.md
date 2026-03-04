# 🏁 F1 Race Finish Predictor

> A machine learning tool that predicts a driver's final race position based on their team and starting grid position.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Models](https://img.shields.io/badge/Models-LogisticRegression%20%7C%20DecisionTree%20%7C%20RandomForest%20%7C%20XGBoost-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🔗 Live App
**[f1-finishposition-predictor.streamlit.app](https://f1-finishposition-predictor.streamlit.app/)**

---

## 🕹️ How it Works

* **Driver & Team Selection:** Choose any driver and constructor from the current grid.
* **Grid Position:** Use the interactive slider to set the starting position.
* **Instant Results:** Generate a predicted finish position based on 2019–2024 race data patterns.

---

## 📊 Data & Logic

- **Dataset:** [Formula 1 World Championship (1950–2020)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) by Rohan Rao on Kaggle — filtered to 2019–2024 race results
- **Models tested:** Logistic Regression, Decision Tree, Random Forest, and XGBoost — best performing model used for predictions
- **Evaluation:** Accuracy score, F1 score, confusion matrix, and classification report
- **Key insight:** Constructor performance plays a significant role alongside grid position in determining final race results

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `Pandas` | Data loading & preprocessing |
| `Scikit-learn` | Model pipeline, encoding & evaluation |
| `LogisticRegression` | Baseline classification model |
| `DecisionTreeClassifier` | Tree-based classification |
| `RandomForestClassifier` | Ensemble classification |
| `XGBClassifier` | Gradient boosted classification |
| `Matplotlib / Seaborn` | Data visualisation |
| `Streamlit` | Web app interface |

---

## 🚀 Run Locally
```bash
git clone https://github.com/subhankar-pattanaik/AI_project.git
cd AI_project
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure
```
AI_project/
├── .gitignore
├── README.md
├── app.py                      # Streamlit app
├── f1_position_predictor_.py   # ML model logic
├── model.css                   # Custom styling
├── requirements.txt
└── data/
    ├── constructors.csv
    ├── driver_standings.csv
    ├── drivers.csv
    ├── qualifying.csv
    ├── races.csv
    └── results.csv
```

---

## 🙏 Acknowledgements

- Race data from [Formula 1 World Championship (1950–2020)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) by Rohan Rao
- Inspired by the Formula 1 community and open-source F1 analytics projects

---

*⚠️ Disclaimer: This project is unofficial and not affiliated with Formula 1 or the FIA.*
