<div align="center">
  <h1>🎓 Student Placement Predictor</h1>
  <p>An End-to-End Classical Machine Learning Pipeline for MBA Student Placement & Salary Prediction</p>
  
  <p>
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
    <img src="https://img.shields.io/badge/XGBoost-%23136200.svg?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost" />
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
    <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  </p>
</div>

<br />

## 📖 About The Project

Recruitment outcomes are heavily determined by academic tracks, past work experiences, and soft skills. Both educational institutions and students struggle to quantify a student's probability of securing an industry placement based strictly on early mathematical markers. 

**This project bridges that gap.** It is a purely data-driven system built to forecast placement success and salary ranges, proactively identifying at-risk students and managing recruitment expectations.

The pipeline ingests academic history, profiles it through complex deterministic feature-engineering, evaluates it against a trained XGBoost Classifier, and passes successful candidates through a dedicated Random Forest Regressor to map exact market salary boundaries. 

---

## ✨ Key Features

- **Dual Prediction Engine**: Predicts whether a student will be placed (Classification) *and* their potential market salary (Regression).
- **Explainable AI (XAI)**: Strips "black box" logic by rendering the top internal model factors (e.g., Performance Trends) that mathematically drove the final prediction.
- **Dynamic Sensitivity Dashboards**: Allows users to interactively drag sliders to see how an extra 5% on their High School exam would mathematically alter their current placement probability today.
- **Robust Feature Engineering**: Replaces weak independent scoring models with Compound Metrics (Consistency Scores, Academic Indexes, and Deterministic Simulations).
- **Modern UI**: Houses the robust AI logic underneath a fully responsive, single-screen **Streamlit** Web Interface.

---

## 🧠 Data & Model Performance

The model trains across anonymized MBA student datasets, heavily prioritized through mathematical transforms. All stochastic/random logic has been eradicated from the baseline pipeline to guarantee that single-user inference results perfectly match trained algorithmic alignments. 

### Best Classification (Placement Target)
By assigning robust class-weight heuristics against minority targets, **XGBoost** captures heavy predictive success:
* **F1-Score**: ~90%
* **ROC-AUC**: ~93%
* **Precision/Recall Equilibrium**: Capable of flagging 93% of successful students.

### Best Regression (Salary Target)
* **Random Forest Regressor** isolates only successfully placed students to predict standard salary bandwidths mapping tightly to market medians.

---

## 💻 Local Setup & Installation

To run the machine learning pipelines and the Streamlit dashboard on your local machine, follow these steps:

**1. Clone the repository**
```bash
git clone https://github.com/dhruvg308/Student_Placement_Predictor.git
cd Student_Placement_Predictor
```

**2. Install all required dependencies**
```bash
pip install -r requirements.txt
```

**3. (Optional) Retrain the core AI models locally**
```bash
# Retrain classification and regression collectively
python train.py
```

**4. Launch the Web Application**
```bash
python -m streamlit run app.py
```

---

## 📂 Project Structure

```text
├── data/
│   └── Placement_Data_Full_Class.csv   # Raw Data Directory
├── saved_models/
│   ├── placement_model.pkl             # Trained Placement Classifier Pipeline
│   └── salary_model.pkl                # Trained Market Salary Regressor Pipeline
├── eda_plots/                          # Exported Visualization Graphics
├── train.py                            # Primary script for data cleaning, engineering, and model training
└── app.py                              # Streamlit Dashboard UI and real-time inference logic
```

---

<div align="center">
  <h3>🛠️ Conceptualized & Built By <a href="https://github.com/your-username">Dhruv Gupta</a></h3>
  <br>
  <i>"Predicting Success. Engineering Futures."</i>
</div>

