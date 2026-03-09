import streamlit as st
import pandas as pd
import xgboost as xgb
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import openpyxl 

# --- 1. PERFORMANCE CACHING (The Speed Fix) ---
@st.cache_data
def load_and_clean_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_resource
def train_clinical_models(data):
    # Stroke Model
    X_s = data.drop("stroke", axis=1)
    y_s = data["stroke"]
    m_s = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42)
    m_s.fit(X_s, y_s)
    
    # Heart Model
    X_h = data.drop("heart_disease", axis=1)
    y_h = data["heart_disease"]
    m_h = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42)
    m_h.fit(X_h, y_h)
    return m_s, m_h

# --- 2. DATABASE ENGINE ---
def save_to_hospital_database(record):
    db_file = "patient_records.xlsx"
    if not os.path.exists(db_file):
        df = pd.DataFrame(columns=record.keys())
        df.to_excel(db_file, index=False)
    
    existing_df = pd.read_excel(db_file)
    updated_df = pd.concat([existing_df, pd.DataFrame([record])], ignore_index=True)
    updated_df.to_excel(db_file, index=False)

# --- 3. UI CONFIG ---
st.set_page_config(page_title="St. Michael AI Hospital", page_icon="🏥", layout="wide")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- 4. LOGIN ---
def login():
    st.title("🏥 St. Michael AI Hospital")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        u = st.text_input("Staff ID")
        p = st.text_input("Access Key", type="password")
        if st.button("Authorize"):
            if u == "admin" and p == "hospital123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid Credentials")

# --- 5. MAIN PORTAL ---
def main():
    st.sidebar.title("👨‍⚕️ Portal")
    if st.sidebar.button("Log Out"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("🧠❤️ Clinical Risk & Record Management")

    # Load Data & Models with Spinner
    data = load_and_clean_data("stroke_prediction_dataset_800_rows.csv")
    if data is not None:
        with st.spinner('Optimizing AI Engines...'):
            model_s, model_h = train_clinical_models(data)
    else:
        st.error("Dataset not found!")
        st.stop()

    # Inputs
    st.markdown("### 📋 Patient Entry")
    p_name = st.text_input("Patient Name:")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age:", 1, 120, 50)
        gender = st.selectbox("Gender:", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
        hyper = st.selectbox("Hypertension:", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    with c2:
        glucose = st.number_input("Glucose:", value=100.0)
        bmi = st.number_input("BMI:", value=25.0)
        smoke = st.selectbox("Smoking:", [0, 1, 2], format_func=lambda x: ["Never", "Former", "Smokes"][x])
    with c3:
        chol = st.number_input("Cholesterol:", value=200.0)
        sys_bp = st.number_input("Systolic:", value=120.0)
        dia_bp = st.number_input("Diastolic:", value=80.0)

    if st.button("RUN SCAN & ARCHIVE"):
        # Logic
        in_s = pd.DataFrame([[age, gender, hyper, 0, glucose, bmi, smoke, chol, sys_bp, dia_bp, 1, 0]], columns=data.drop("stroke", axis=1).columns)
        s_prob = float(model_s.predict_proba(in_s)[0][1])
        
        in_h = pd.DataFrame([[age, gender, hyper, glucose, bmi, smoke, chol, sys_bp, dia_bp, 1, 0, (1 if s_prob > 0.5 else 0)]], columns=data.drop("heart_disease", axis=1).columns)
        h_prob = float(model_h.predict_proba(in_h)[0][1])

        # Sensitive Threshold (0.30)
        def get_risk(p):
            if p > 0.7: return "High Risk Detected", "error"
            if p > 0.3: return "Likely to have soon", "warning"
            return "Low Risk", "success"

        s_stat, s_type = get_risk(s_prob)
        h_stat, h_type = get_risk(h_prob)

        # Results
        st.divider()
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"### 🧠 Stroke: {s_stat}")
            if s_prob > 0.3: st.info("The person should go see a doctor for precautionary measures.")
        with r2:
            st.markdown(f"### ❤️ Heart: {h_stat}")
            if h_prob > 0.3: st.info("The person should go see a doctor for precautionary measures.")

        # ARCHIVE RECORD
        record = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Patient": p_name if p_name else "Anonymous",
            "Stroke_Status": s_stat,
            "Heart_Status": h_stat,
            "Glucose": glucose,
            "BMI": bmi
        }
        save_to_hospital_database(record)
        st.success("💾 Record captured and archived in the terminal database.")

    # ARCHIVE VIEW
    st.divider()
    st.subheader("📂 Hospital Archives")
    if os.path.exists("patient_records.xlsx"):
        history = pd.read_excel("patient_records.xlsx")
        st.dataframe(history.iloc[::-1], use_container_width=True)

# Routing
if not st.session_state.logged_in:
    login()
else:
    main()
