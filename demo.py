import streamlit as st
import pandas as pd
import xgboost as xgb
import os

# 1. Page Config
st.set_page_config(page_title="St. Michael AI Hospital", page_icon="🏥", layout="wide")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- LOGIN SCREEN ---
def login_screen():
    st.title("🏥 St. Michael AI Hospital")
    st.subheader("Dual Stroke & Heart Diagnostic System")
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

    st.title("🧠❤️ Dual AI Diagnostic Portal")
    
    file_path = "stroke_prediction_dataset_800_rows.csv"
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        X_stroke = data.drop("stroke", axis=1)
        y_stroke = data["stroke"]
        model_stroke = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
        model_stroke.fit(X_stroke, y_stroke)
        
        X_heart = data.drop("heart_disease", axis=1)
        y_heart = data["heart_disease"]
        model_heart = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
        model_heart.fit(X_heart, y_heart)
    else:
        st.error("❌ Dataset missing! Ensure the CSV is on GitHub.")
        st.stop()

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

    if st.button("RUN COMPREHENSIVE DIAGNOSTIC", use_container_width=True):
        st.divider()
        
        # Calculations
        p_data_stroke = pd.DataFrame([[age, gender, hyper, 0, glucose, bmi, smoke, chol, sys_bp, dia_bp, 1, 0]], columns=X_stroke.columns)
        stroke_score = model_stroke.predict_proba(p_data_stroke)[0][1]
        
        p_data_heart = pd.DataFrame([[age, gender, hyper, glucose, bmi, smoke, chol, sys_bp, dia_bp, 1, 0, (1 if stroke_score > 0.5 else 0)]], columns=X_heart.columns)
        heart_score = model_heart.predict_proba(p_data_heart)[0][1]

        # Results Display with % over 100
        res1, res2 = st.columns(2)
        
        with res1:
            st.subheader("🧠 Stroke Risk Analysis")
            stroke_pct = stroke_score * 100
            st.metric("Risk Level", f"{stroke_pct:.1f}% / 100%")
            st.progress(stroke_score) # Visual bar
            if stroke_pct < 30: st.success("CATEGORY: LOW")
            elif stroke_pct < 70: st.warning("CATEGORY: MODERATE")
            else: st.error("CATEGORY: HIGH")

        with res2:
            st.subheader("❤️ Heart Disease Analysis")
            heart_pct = heart_score * 100
            st.metric("Risk Level", f"{heart_pct:.1f}% / 100%")
            st.progress(heart_score) # Visual bar
            if heart_pct < 30: st.success("CATEGORY: LOW")
            elif heart_pct < 70: st.warning("CATEGORY: MODERATE")
            else: st.error("CATEGORY: HIGH")

        # Visual Chart
        st.divider()
        st.subheader("📊 Patient vs. Global Average")
        avg_data = pd.DataFrame({
            'Metric': ['Glucose', 'BMI', 'Cholesterol'],
            'Patient': [glucose, bmi, chol],
            'Avg': [data['avg_glucose_level'].mean(), data['bmi'].mean(), data['cholesterol'].mean()]
        }).set_index('Metric')
        st.bar_chart(avg_data)

# --- ROUTING ---
if not st.session_state.logged_in:
    login_screen()
else:
    main_app()
