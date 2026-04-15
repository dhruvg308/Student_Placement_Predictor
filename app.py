import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import contextlib
import io

# Import the feature engineering component from our single training script
# This is how students typically share common ML logic
try:
    from train import engineer_features
except ImportError:
    st.error("Could not find train.py! Ensure it's in the same directory.")
    st.stop()

# ── Settings ──
MODELS_DIR = "saved_models"
DROP_COLS = ["sl_no", "salary", "status"]

# ── Page Config & CSS ──
st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 0; }
.result-placed {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: white; text-align: center; padding: 1rem; border-radius: 12px; margin-bottom: 0.8rem;
}
.result-notplaced {
    background: linear-gradient(135deg, #eb3349, #f45c43);
    color: white; text-align: center; padding: 1rem; border-radius: 12px; margin-bottom: 0.8rem;
}
.result-placed h3, .result-notplaced h3 { margin: 0; font-size: 1.4rem; }
.metric-box {
    background: #f4f6ff; border: 1px solid #dde2ff; border-radius: 10px;
    padding: 0.7rem 1rem; text-align: center; margin-bottom: 0.5rem;
}
.metric-box .lbl { font-size: 0.7rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
.metric-box .val { font-size: 1.3rem; font-weight: 800; color: #333; }
.metric-box .val.g { color: #11998e; }
.metric-box .val.b { color: #667eea; }
</style>
""", unsafe_allow_html=True)

# ── Load Model Caches ──
@st.cache_resource
def load_models():
    clf_path = os.path.join(MODELS_DIR, "placement_model.pkl")
    reg_path = os.path.join(MODELS_DIR, "salary_model.pkl")
    if not os.path.exists(clf_path) or not os.path.exists(reg_path):
        raise FileNotFoundError("Missing models! Please run `python train.py` first.")
    
    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)
    return clf, reg

try:
    clf_pipeline, reg_pipeline = load_models()
except Exception as e:
    st.error(str(e))
    st.stop()

# ── Procedural Inference Handlers ──
def predict_candidate(inputs_dict):
    df = pd.DataFrame([inputs_dict])
    df_eng = engineer_features(df)
    
    X_infer = df_eng.copy()
    for col in DROP_COLS:
        if col in X_infer.columns:
            X_infer = X_infer.drop(columns=[col])

    # 1. Classification
    pred = clf_pipeline.predict(X_infer)[0]
    prob = clf_pipeline.predict_proba(X_infer)[0][1] if hasattr(clf_pipeline['classifier'], "predict_proba") else None
    placed = bool(pred)
    
    # 2. Regression (if placed)
    sal_range = None
    if placed:
        base_sal = reg_pipeline.predict(X_infer)[0]
        sal_min = max(200000.0, base_sal * 0.9)
        sal_max = base_sal * 1.1
        sal_range = (sal_min, sal_max)
        
    # 3. Factor Importance Extraction
    importances = clf_pipeline['classifier'].feature_importances_
    names = [name.split('__')[-1] for name in clf_pipeline['preprocessor'].get_feature_names_out()]
    imp_df = pd.DataFrame({'Feature': names, 'Importance': importances}).sort_values('Importance', ascending=False)

    return placed, prob, sal_range, imp_df.head(4).to_dict('records')

def sweep_sensitivity(base_dict, sweep_field, sweep_vals):
    probs = []
    for v in sweep_vals:
        tmp = base_dict.copy()
        tmp[sweep_field] = v
        df = pd.DataFrame([tmp])
        with contextlib.redirect_stdout(io.StringIO()):
            df_eng = engineer_features(df)
            
        X = df_eng.copy()
        for col in DROP_COLS:
            if col in X.columns:
                X = X.drop(columns=[col])
        p = clf_pipeline.predict_proba(X)[0][1]
        probs.append(p)
    return probs

# ── Title ──
st.markdown("## 🎓 Student Placement Predictor")

# ── Frontend Layout ──
c1, c2, c3 = st.columns([1, 1, 1.3], gap="medium")

with c1:
    st.caption("📚 ACADEMICS")
    a1, a2 = st.columns(2)
    ssc_p = a1.number_input("10th %", 0.0, 100.0, 72.0, 0.5)
    hsc_p = a2.number_input("12th %", 0.0, 100.0, 68.0, 0.5)
    a3, a4 = st.columns(2)
    degree_p = a3.number_input("Degree %", 0.0, 100.0, 66.0, 0.5)
    mba_p = a4.number_input("MBA %", 0.0, 100.0, 62.0, 0.5)
    a5, a6 = st.columns(2)
    ssc_b = a5.selectbox("10th Board", ["Central", "Others"])
    hsc_b = a6.selectbox("12th Board", ["Central", "Others"])
    hsc_s = st.selectbox("12th Stream", ["Science", "Commerce", "Arts"])

with c2:
    st.caption("👤 PROFILE")
    p1, p2 = st.columns(2)
    gender = p1.selectbox("Gender", ["M", "F"])
    workex = p2.selectbox("Work Exp", ["No", "Yes"])
    degree_t = st.selectbox("Degree Type", ["Sci&Tech", "Comm&Mgmt", "Others"])
    specialisation = st.selectbox("MBA Spec.", ["Mkt&Fin", "Mkt&HR"])
    etest_p = st.slider("Employability Test %", 0.0, 100.0, 70.0, 1.0)
    
    st.markdown("")
    if st.button("🔮 Predict", type="primary", use_container_width=True):
        st.session_state['predicted'] = True

with c3:
    st.caption("📊 RESULTS")
    if st.session_state.get('predicted', False):
        raw_input = {
            'gender': gender, 'ssc_p': ssc_p, 'ssc_b': ssc_b,
            'hsc_p': hsc_p, 'hsc_b': hsc_b, 'hsc_s': hsc_s,
            'degree_p': degree_p, 'degree_t': degree_t,
            'workex': workex, 'etest_p': etest_p,
            'specialisation': specialisation, 'mba_p': mba_p
        }

        # Run procedural inference
        placed, prob, sal_range, factors = predict_candidate(raw_input)

        # Draw Outcome
        if placed:
            st.markdown('<div class="result-placed"><h3>✅ PLACED</h3></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-notplaced"><h3>❌ NOT PLACED</h3></div>', unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        prob_str = f"{prob*100:.1f}%" if prob else "N/A"
        clr = "g" if prob and prob >= 0.5 else ""
        m1.markdown(f'<div class="metric-box"><div class="lbl">Probability</div><div class="val {clr}">{prob_str}</div></div>', unsafe_allow_html=True)

        sal_str = f"₹{sal_range[0]/100000:.1f}L – ₹{sal_range[1]/100000:.1f}L" if sal_range else "N/A"
        m2.markdown(f'<div class="metric-box"><div class="lbl">Salary Range</div><div class="val b">{sal_str}</div></div>', unsafe_allow_html=True)

        # Draw Factors
        st.caption("🏆 TOP FACTORS")
        max_imp = max(f['Importance'] for f in factors)
        for f in factors[:3]:
            pct = f['Importance'] / max_imp
            name = f['Feature'].replace('_', ' ').title()
            st.progress(pct, text=f"{name} — {f['Importance']:.3f}")

        # Draw Sensitivity
        st.caption("📈 SENSITIVITY")
        swp = st.selectbox("Sweep", ["ssc_p", "hsc_p", "degree_p", "mba_p", "etest_p"],
            format_func=lambda x: {"ssc_p":"10th %","hsc_p":"12th %","degree_p":"Degree %","mba_p":"MBA %","etest_p":"E-Test"}[x])
        vals = np.arange(30, 101, 5).tolist()
        probs = sweep_sensitivity(raw_input, swp, vals)
        st.line_chart(pd.DataFrame({"Score": vals, "Prob": probs}).set_index("Score"), height=180)
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#bbb;">
            <p style="font-size:2.5rem;">🎯</p>
            <p>Fill in details and click <b>Predict</b></p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("---")
st.markdown("<div style='text-align: center; color: #888; font-size: 0.9rem; padding: 1rem 0 2rem 0;'>Made By Dhruv Gupta</div>", unsafe_allow_html=True)
