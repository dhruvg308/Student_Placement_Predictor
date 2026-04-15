import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. CONSTANTS & CONFIGURATION
# ==========================================
RAW_DATA_PATH = "data/Placement_Data_Full_Class.csv"
MODELS_DIR = "saved_models"
TARGET_COL = "status"
DROP_COLS = ["sl_no", "salary"]

# Features we expect after engineering
NUMERIC_FEATURES = [
    "ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p",
    "academic_index", "consistency_score", "performance_trend",
    "employability_score", "skill_proxy", "industry_demand",
    "sim_aptitude", "sim_communication", "sim_coding", "sim_interview"
]
CATEGORICAL_FEATURES = [
    "gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", 
    "workex", "specialisation", "college_tier"
]

os.makedirs(MODELS_DIR, exist_ok=True)

# ==========================================
# 2. FEATURE ENGINEERING FUNCTION
# ==========================================
def engineer_features(df):
    """
    Performs fully mathematical, deterministic feature engineering.
    """
    df_copy = df.copy()
    
    # ── Academic Computations ──
    df_copy['academic_index'] = df_copy[['ssc_p', 'hsc_p', 'degree_p', 'mba_p']].mean(axis=1)
    std_devs = df_copy[['ssc_p', 'hsc_p', 'degree_p', 'mba_p']].std(axis=1)
    df_copy['consistency_score'] = 100 / (std_devs + 1)
    df_copy['performance_trend'] = df_copy['mba_p'] - df_copy['ssc_p']
    
    # ── Employability & Demand Computations ──
    workex_bonus = df_copy['workex'].map({'Yes': 10, 'No': 0}).fillna(0)
    df_copy['employability_score'] = (df_copy['academic_index'] * 0.4) + (df_copy['etest_p'] * 0.4) + workex_bonus
    df_copy['skill_proxy'] = (df_copy['degree_p'] * 0.6) + (df_copy['mba_p'] * 0.4)
    demand_mapping = {'Mkt&Fin': 85, 'Mkt&HR': 70}
    df_copy['industry_demand'] = df_copy['specialisation'].map(demand_mapping).fillna(60)
    
    # College tier check safely
    if 'ssc_b' in df_copy.columns and 'hsc_b' in df_copy.columns:
        def get_tier(row):
            score = (row['ssc_b'] == 'Central') + (row['hsc_b'] == 'Central')
            if score == 2: return 'Tier 1'
            elif score == 1: return 'Tier 2'
            else: return 'Tier 3'
        df_copy['college_tier'] = df_copy.apply(get_tier, axis=1)
    else:
        df_copy['college_tier'] = 'Tier 2'
        
    # ── Simulated Scores (Deterministic Mathematics) ──
    df_copy['sim_aptitude'] = (df_copy['etest_p'] * 0.7 + df_copy['academic_index'] * 0.3).clip(0, 100)
    com_bonus = df_copy['specialisation'].map({'Mkt&HR': 5, 'Mkt&Fin': 0}).fillna(0)
    df_copy['sim_communication'] = (df_copy['mba_p'] * 0.5 + df_copy['hsc_p'] * 0.2 + 20 + com_bonus).clip(0, 100)
    tech_bonus = df_copy['degree_t'].map({'Sci&Tech': 15, 'Comm&Mgmt': 0, 'Others': -5}).fillna(0)
    df_copy['sim_coding'] = (df_copy['degree_p'] * 0.5 + df_copy['etest_p'] * 0.2 + tech_bonus + 10).clip(0, 100)
    df_copy['sim_interview'] = (
        df_copy['sim_communication'] * 0.35 +
        df_copy['sim_aptitude'] * 0.35 +
        df_copy['sim_coding'] * 0.2 +
        df_copy['performance_trend'] * 0.1 + 15
    ).clip(0, 100)

    return df_copy

# ==========================================
# 3. GET PREPROCESSOR PIPELINE
# ==========================================
def get_preprocessor():
    # Defines standard scaler and one-hot encoding logic
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])
    return preprocessor

# ==========================================
# 4. TRAINING LOGIC
# ==========================================
if __name__ == "__main__":
    print("[RUNNING] Starting Student Placement ML Training Pipeline...")

    # Load Database
    print(f"[1/4] Loading and cleaning {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Generic Dataset Cleaning
    df = df.drop_duplicates()
    if 'salary' in df.columns:
        df['salary'] = df['salary'].fillna(0.0)
    for col in ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']:
        if col in df.columns:
            df[col] = df[col].clip(0.0, 100.0)
            
    # Fix category labels securely
    for cat_col in ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status']:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str).str.strip()
    
    # Map Placed/Not Placed target directly to 1 and 0 classification formats
    if 'status' in df.columns and df['status'].dtype == object:
        df['status'] = df['status'].map({'Placed': 1, 'Not Placed': 0})
        
    print(f"Data cleaned. Proceeding with {df.shape[0]} valid records.")

    # Apply Features
    print("[2/4] Executing feature engineering logic...")
    df = engineer_features(df)
    
    # ──────────────────────────────────────────────────────────
    # A. TRAIN CLASSIFICATION MODEL (WILL THE STUDENT PLACE?) 
    # ──────────────────────────────────────────────────────────
    print("\n================ CLASSIFICATION ================")
    X = df.drop(columns=[TARGET_COL] + DROP_COLS)
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training split generated -> Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', scale_pos_weight=0.5)
    }
    
    best_clf_score = 0
    best_clf_name = None
    best_clf_pipeline = None

    preprocessor = get_preprocessor()

    print("\nBenchmarking classification estimators...")
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        f1 = f1_score(y_test, y_pred, zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
        
        print(f" - [{name}] F1-Score: {f1:.4f} | Accuracy: {acc:.4f} | ROC-AUC: {roc:.4f}")
        
        if f1 > best_clf_score:
            best_clf_score = f1
            best_clf_name = name
            best_clf_pipeline = pipeline

    clf_path = os.path.join(MODELS_DIR, "placement_model.pkl")
    joblib.dump(best_clf_pipeline, clf_path)
    print(f"\n[WINNER] -> Saved '{best_clf_name}' as classification core: {clf_path}")

    # ──────────────────────────────────────────────────────────
    # B. TRAIN REGRESSION MODEL (HOW MUCH SALARY IF PLACED?)
    # ──────────────────────────────────────────────────────────
    print("\n================ REGRESSION ================")
    # Salary regression model strictly on placed students
    df_placed = df[(df['status'] == 1) & (df['salary'].notnull()) & (df['salary'] > 0)].copy()
    
    if len(df_placed) > 0:
        X_reg = df_placed.drop(columns=[TARGET_COL, 'salary'] + [c for c in DROP_COLS if c != 'salary'])
        y_reg = df_placed['salary']
        
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        print(f"Regression split generated using Placed students -> Train: {Xr_train.shape[0]} | Test: {Xr_test.shape[0]}")
        
        reg_models = {
            'Linear Regression': LinearRegression(),
            'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        best_reg_score = -float('inf')  # Use R2 for regression
        best_reg_name = None
        best_reg_pipeline = None
        
        print("\nBenchmarking regression estimators...")
        for name, rmodel in reg_models.items():
            r_pipe = Pipeline(steps=[('preprocessor', get_preprocessor()), ('regressor', rmodel)])
            r_pipe.fit(Xr_train, yr_train)
            yr_pred = r_pipe.predict(Xr_test)
            
            mae = mean_absolute_error(yr_test, yr_pred)
            r2 = r2_score(yr_test, yr_pred)
            print(f" - [{name}] MAE: Rs {mae:,.0f} | R2 Score: {r2:.4f}")
            
            if r2 > best_reg_score:
                best_reg_score = r2
                best_reg_name = name
                best_reg_pipeline = r_pipe
                
        reg_path = os.path.join(MODELS_DIR, "salary_model.pkl")
        joblib.dump(best_reg_pipeline, reg_path)
        print(f"\n[WINNER] -> Saved '{best_reg_name}' as regression core: {reg_path}")
    else:
        print("Not enough placed records to train the regression component.")

    print("\n[COMPLETE] [4/4] Entire Machine Learning phase complete!")
