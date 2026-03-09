import streamlit as st
import pandas as pd
import xgboost as xgb
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import openpyxl 

# --- 1. Page Configuration & Style ---
st.set_page_config(page_title="TEAM A HOSPITAL", page_icon="🏥", layout="wide")

# Custom CSS for a cleaner UX
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #004b95; color: white; }
    </style>
    """, unsafe_allow_html=True)

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- 2. Database Engine ---
def save_to_excel(new_data):
    db_file = "patient_records.xlsx"
    if not os.path.exists(db_file):
        df = pd.DataFrame(columns=new_data.keys())
        df.to_excel(db_file, index=False)
    existing_df = pd.read_excel(db_file)
    updated_df = pd.concat([existing_df, pd.DataFrame([new_data])], ignore_index=True)
    updated_df.to_excel(db_file, index=False)

# --- 3. Login Screen ---
def login_screen():
    st.title("🏥 TEAM A HOSPITAL")
    st.subheader("Dual Stroke & Heart Diagnostic System ")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        user = st.text_input("Doctor ID")
        pw = st.text_input("Access Key", type="password")
        if st.button("Authorize Access"):
            if user == "admin" and pw == "hospital123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid Credentials")

# --- 4. Main App ---
def main_app():
    st.sidebar.title("👨‍⚕️ Clinical Workspace")
    st.sidebar.write("Mode: **High Sensitivity**")
    if st.sidebar.button("Log Out"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("🧠❤️ Clinical Risk & Record Portal")
    
    # Load Training Data
    file_path = "stroke_prediction_dataset_800_rows.csv"
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        # Stroke Model
        X_s = data.drop("stroke", axis=1)
        y_s = data["stroke"]
        model_s = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
        model_s.fit(X_s, y_s)
        # Heart Model
        X_h = data.drop("heart_disease", axis=1)
        y_h = data["heart_disease"]
        model_h = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
        model_h.fit(X_h, y_h)
    else:
        st.error("❌ Dataset missing!")
        st.stop()

    # --- UI: Input Panel ---
    st.markdown("### 📋 Patient Clinical Profile")
    p_name = st.text_input("Patient Name / ID:")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age:", 1, 120, 50)
        gender = st.selectbox("Gender:", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
        hyper = st.selectbox("Hypertension:", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    with c2:
        glucose = st.number_input("Avg Glucose Level:", value=110.0)
        bmi = st.number_input("BMI:", value=28.0)
        smoke = st.selectbox("Smoking Status:", [0, 1, 2], format_func=lambda x: ["Never", "Former", "Smokes"][x])
    with c3:
        chol = st.number_input("Cholesterol:", value=200.0)
        sys_bp = st.number_input("Systolic BP:", value=120.0)
        dia_bp = st.number_input("Diastolic BP:", value=80.0)

    # --- Logic: Diagnosis ---
    if st.button("EXECUTE CLINICAL SCAN"):
        st.divider()
        
        # Prepare inputs
        p_input_s = pd.DataFrame([[age, gender, hyper, 0, glucose, bmi, smoke, chol, sys_bp, dia_bp, 1, 0]], columns=X_s.columns)
        s_prob = float(model_s.predict_proba(p_input_s)[0][1])
        
        p_input_h = pd.DataFrame([[age, gender, hyper, glucose, bmi, smoke, chol, sys_bp, dia_bp, 1, 0, (1 if s_prob > 0.5 else 0)]], columns=X_h.columns)
        h_prob = float(model_h.predict_proba(p_input_h)[0][1])

        # UX: SENSITIVE THRESHOLD (0.30)
        threshold = 0.30
        
        # Determine Status
        def get_status(p):
            if p > 0.70: return "High Risk Detected", "error"
            if p > threshold: return "Likely to have soon", "warning"
            return "Low Risk", "success"

        s_status, s_type = get_status(s_prob)
        h_status, h_type = get_status(h_prob)

        # UI: Results Row
        res1, res2 = st.columns(2)
        with res1:
            st.subheader("🧠 Stroke Assessment")
            if s_type == "error": st.error(f"**{s_status}**")
            elif s_type == "warning": st.warning(f"**{s_status}**")
            else: st.success(f"**{s_status}**")
            if s_prob > threshold: st.info("Advice: Go see a doctor for precautionary measures.")

        with res2:
            st.subheader("❤️ Heart Assessment")
            if h_type == "error": st.error(f"**{h_status}**")
            elif h_type == "warning": st.warning(f"**{h_status}**")
            else: st.success(f"**{h_status}**")
            if h_prob > threshold: st.info("Advice: Go see a doctor for precautionary measures.")

        # UI: Raw Probability Table (UX from your sample)
        st.subheader("📊 Analytical Probability Breakdown")
        probs_df = pd.DataFrame({
            "Condition": ["Cerebrovascular (Stroke)", "Cardiovascular (Heart)"],
            "Probability Index": [f"{s_prob:.1%}", f"{h_prob:.1%}"],
            "Threshold Check": ["ABOVE LIMIT" if s_prob > threshold else "SAFE", 
                               "ABOVE LIMIT" if h_prob > threshold else "SAFE"]
        })
        st.table(probs_df)

        # UI: Feature Importance Chart (Visual UX)
        st.subheader("📉 Primary Risk Driver Analysis")
        # Combine importances for a summary
        imp = pd.Series(model_s.feature_importances_, index=X_s.columns).sort_values(ascending=False).head(5)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=imp.values, y=imp.index, palette="RdYlGn_r", ax=ax)
        st.pyplot(fig)

        # Save Record
        save_to_excel({
            "Time": datetime.now().strftime("%H:%M"), "Patient": p_name, "Age": age,
            "Stroke": s_status, "Heart": h_status
        })
        st.toast(f"Record for {p_name} saved!")

    # UI: Archives Table
    st.divider()
    st.subheader("📂 Hospital Records Archive")
    if os.path.exists("patient_records.xlsx"):
        history = pd.read_excel("patient_records.xlsx")
        st.dataframe(history.iloc[::-1], use_container_width=True)

# --- Routing ---
if not st.session_state.logged_in:
    login_screen()
else:
    main_app()

