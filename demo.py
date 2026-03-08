import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import os

# 1. Page Config
st.set_page_config(page_title="St. Michael Cardio-Portal", page_icon="🏥", layout="wide")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- LOGIN SCREEN ---
def login_screen():
    st.title("🏥 St. Michael AI Hospital")
    st.subheader("Dual Stroke & Heart Disease Diagnostic Portal")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        user = st.text_input("Staff ID")
        pw = st.text_input("Access Key", type="password")
        if st.button("Authorize Access", use_container_width=True):
            if user == "admin" and pw == "hospital123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid Credentials")

# --- MAIN APP ---
def main_app():
    st.sidebar.title("👨‍⚕️ Staff Portal")
    if st.sidebar.button("Log Out"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("🧠❤️ Dual AI Diagnostic Engine")
    
    file_path = "stroke_prediction_dataset_800_rows.csv"
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        
        # MODEL 1: STROKE DETECTION
        X_stroke = data.drop("stroke", axis=1)
        y_stroke = data["stroke"]
        model_stroke = XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
        model_stroke.fit(X_stroke, y_stroke)
        
        # MODEL 2: HEART DISEASE DETECTION
        X_heart = data.drop("heart_disease", axis=1)
        y_heart = data["heart_disease"]
        model_heart = XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
        model_heart.fit(X_heart, y_heart)
        
        st.success(f"✅ Dual-Engine Synchronized: {len(data)} clinical records processed.")
    else:
        st.error("❌ Dataset missing! Upload 'stroke_prediction_dataset_800_rows.csv' to GitHub.")
        st.stop()

    # INPUT FIELDS
    st.markdown("### Patient Vitals Entry")
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

    # Missing indicators for specific predictions
    act = 1 # Average Activity
    alc = 0 # No Alcohol
    heart_disease_status = 0 # Placeholder for Stroke model

    if st.button("RUN DUAL DIAGNOSTIC SCAN", use_container_width=True):
        st.divider()
        col_left, col_right = st.columns(2)
        
        # 1️⃣ STROKE PREDICTION
        p_stroke = pd.DataFrame([[age, gender, hyper, heart_disease_status, glucose, bmi, smoke, chol, sys_bp, dia_bp, act, alc]], columns=X_stroke.columns)
        stroke_score = model_stroke.predict_proba(p_stroke)[0][1]
        
        with col_left:
            st.subheader("🧠 Stroke Assessment")
            if stroke_score < 0.30: st.success(f"LOW RISK ({stroke_score:.1%})")
            elif stroke_score < 0.70: st.warning(f"MODERATE RISK ({stroke_score:.1%})")
            else: st.error(f"HIGH RISK ({stroke_score:.1%})")

        # 2️⃣ HEART DISEASE PREDICTION
        # Note: We use the 'stroke' prediction as an input for heart disease if needed, or just standard vitals
        stroke_status = 1 if stroke_score > 0.5 else 0
        p_heart = pd.DataFrame([[age, gender, hyper, glucose, bmi, smoke, chol, sys_bp, dia_bp, act, alc, stroke_status]], columns=X_heart.columns)
        heart_score = model_heart.predict_proba(p_heart)[0][1]
        
        with col_right:
            st.subheader("❤️ Heart Assessment")
            if heart_score < 0.30: st.success(f"LOW RISK ({heart_score:.1%})")
            elif heart_score < 0.70: st.warning(f"MODERATE RISK ({heart_score:.1%})")
            else: st.error(f"HIGH RISK ({heart_score:.1%})")

        # GENERATE CONSOLIDATED REPORT
        report = f"ST. MICHAEL DUAL CLINICAL REPORT\n" + "-"*30 + \
                 f"\nStroke Risk: {stroke_score:.1%}\nHeart Risk: {heart_score:.1%}"
        st.download_button("📄 Download Consolidated Report", report, "clinical_report.txt")

# --- NAVIGATION ---
if not st.session_state.logged_in:
    login_screen()
else:
    main_app()
