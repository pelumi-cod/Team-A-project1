import streamlit as st
import pandas as pd
import xgboost as xgb
import os

# 1. Page Configuration (Hospital Brand)
st.set_page_config(page_title="St. Michael AI Hospital", page_icon="🏥", layout="wide")

# Initialize Session State for Login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- LOGIN SCREEN ---
def login_screen():
    st.title("🏥 St. Michael AI Hospital")
    st.subheader("Dual Stroke & Heart Diagnostic System")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("Medical Staff Authentication Required")
        user = st.text_input("Doctor ID")
        pw = st.text_input("Access Key", type="password")
        if st.button("Authorize Access", use_container_width=True):
            if user == "admin" and pw == "hospital123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid Credentials")

# --- MAIN CLINICAL PORTAL ---
def main_app():
    st.sidebar.title("👨‍⚕️ Staff Portal")
    st.sidebar.info("System: Active\nEngine: XGBoost v2.0")
    if st.sidebar.button("Log Out"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("🧠❤️ Clinical Risk Assessment Portal")
    
    # LOAD AND TRAIN (800-Row Dataset)
    file_path = "stroke_prediction_dataset_800_rows.csv"
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        
        # Stroke Engine
        X_s = data.drop("stroke", axis=1)
        y_s = data["stroke"]
        model_s = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
        model_s.fit(X_s, y_s)
        
        # Heart Engine
        X_h = data.drop("heart_disease", axis=1)
        y_h = data["heart_disease"]
        model_h = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
        model_h.fit(X_h, y_h)
        
        st.success(f"✅ AI Engines Synchronized with {len(data)} clinical records.")
    else:
        st.error("❌ Dataset missing! Please upload the CSV to your GitHub.")
        st.stop()

    # INPUT PANEL (3-Column Layout)
    st.markdown("### 📋 Patient Vitals Entry")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age:", 1.0, 120.0, 50.0)
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

    # ANALYSIS LOGIC
    if st.button("RUN FULL CLINICAL DIAGNOSIS", use_container_width=True):
        st.divider()
        
        # Prepare Data
        p_stroke = pd.DataFrame([[age, gender, hyper, 0, glucose, bmi, smoke, chol, sys_bp, dia_bp, 1, 0]], columns=X_s.columns)
        s_score = float(model_s.predict_proba(p_stroke)[0][1])
        
        p_heart = pd.DataFrame([[age, gender, hyper, glucose, bmi, smoke, chol, sys_bp, dia_bp, 1, 0, (1 if s_score > 0.5 else 0)]], columns=X_h.columns)
        h_score = float(model_h.predict_proba(p_heart)[0][1])

        # RESULTS (No percentages, just your words)
        res1, res2 = st.columns(2)
        
        with res1:
            st.markdown("### 🧠 Stroke Assessment")
            if s_score < 0.3:
                st.success("### STATUS: Low Risk")
                st.write("Patient clinical markers are currently stable.")
            elif s_score < 0.7:
                st.warning("### STATUS: Likely to have soon")
                st.info("⚠️ **Action Required:** The person should go see a doctor for precautionary measures.")
            else:
                st.error("### STATUS: High Risk Detected")
                st.write("Immediate medical intervention is highly recommended.")

        with res2:
            st.markdown("### ❤️ Heart Assessment")
            if h_score < 0.3:
                st.success("### STATUS: Low Risk")
                st.write("Cardiovascular metrics are within normal range.")
            elif h_score < 0.7:
                st.warning("### STATUS: Likely to have soon")
                st.info("⚠️ **Action Required:** The person should go see a doctor for precautionary measures.")
            else:
                st.error("### STATUS: High Risk Detected")
                st.write("Critical cardiovascular abnormalities detected.")

        # DATA ANALYTICS (The Chart)
        st.divider()
        st.subheader("📊 Clinical Data Comparison")
        avg_vals = [data['avg_glucose_level'].mean(), data['bmi'].mean(), data['cholesterol'].mean()]
        chart_df = pd.DataFrame({
            'Metric': ['Glucose', 'BMI', 'Cholesterol'],
            'Current Patient': [glucose, bmi, chol],
            'Global Average': avg_vals
        }).set_index('Metric')
        st.bar_chart(chart_df)

        # PRINT/DOWNLOAD SESSION
        report = f"ST. MICHAEL CLINICAL REPORT\n----------------\nAge: {age}\nStroke Status: {'High' if s_score > 0.7 else 'Likely' if s_score > 0.3 else 'Low'}\nHeart Status: {'High' if h_score > 0.7 else 'Likely' if h_score > 0.3 else 'Low'}"
        st.download_button("📄 Download/Print Report", report, "patient_report.txt", use_container_width=True)

# --- RUNTIME ---
if not st.session_state.logged_in:
    login_screen()
else:
    main_app()
