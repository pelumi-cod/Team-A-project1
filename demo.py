import streamlit as st
import pandas as pd
import xgboost as xgb
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import openpyxl 

# --- 1. PERFORMANCE CACHING ---
@st.cache_data
def load_and_clean_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_resource
def train_clinical_models(data):
    X_s = data.drop("stroke", axis=1)
    y_s = data["stroke"]
    m_s = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42)
    m_s.fit(X_s, y_s)
    
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
        glucose = st.number_input("Glucose:", value=110.0)
        bmi = st.number_input("BMI:", value=28.0)
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

        def get_risk_label(p):
            if p > 0.7: return "High Risk Detected"
            if p > 0.3: return "Likely to have soon"
            return "Low Risk"

        s_status = get_risk_label(s_prob)
        h_status = get_risk_label(h_prob)

        # Show Results
        st.divider()
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"### 🧠 Stroke: {s_status}")
        with r2:
            st.markdown(f"### ❤️ Heart: {h_status}")

        # SAVE RECORD
        record = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Patient Name": p_name if p_name else "Guest Patient",
            "Age": age,
            "Stroke Assessment": s_status,
            "Heart Assessment": h_status,
            "Glucose": glucose,
            "BMI": bmi
        }
        save_to_hospital_database(record)
        
        # DISPLAY GRAPH
        st.subheader("📊 Primary Risk Factors")
        imp = pd.Series(model_s.feature_importances_, index=data.drop("stroke", axis=1).columns).sort_values(ascending=False).head(5)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=imp.values, y=imp.index, palette="viridis", ax=ax)
        st.pyplot(fig)

    # --- THE FIXED TABLE VIEW ---
    st.divider()
    st.subheader("📂 Hospital Archives")
    if os.path.exists("patient_records.xlsx"):
        history = pd.read_excel("patient_records.xlsx")
        
        # This line forces the table to ONLY show these columns and hides the mess
        display_cols = ["Timestamp", "Patient Name", "Age", "Stroke Assessment", "Heart Assessment", "Glucose", "BMI"]
        
        # We check which of our desired columns actually exist in the file
        available_cols = [c for c in display_cols if c in history.columns]
        
        # Show only the clean version
        st.dataframe(history[available_cols].iloc[::-1], use_container_width=True)
    else:
        st.info("No records to display.")

# Routing
if not st.session_state.logged_in:
    login()
else:
    main()

