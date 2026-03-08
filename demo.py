import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import os

# --- 1. Page Configuration ---
st.set_page_config(page_title="St. Michael AI Hospital", page_icon="🏥", layout="wide")

# Initialize Login State
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- 2. THE LOGIN SCREEN ---
def login_screen():
    st.title("🏥 St. Michael AI Hospital")
    st.markdown("### Clinical Decision Support System")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("Authorized Medical Personnel Only")
        username = st.text_input("Staff ID")
        password = st.text_input("Access Key", type="password")
        
        if st.button("Authorize Access", use_container_width=True):
            # You can change these credentials as needed
            if username == "admin" and password == "hospital123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid Credentials. Please check your Staff ID or Key.")

# --- 3. THE MAIN DIAGNOSTIC APP ---
def main_app():
    # Sidebar for Logout and User Info
    st.sidebar.title("👨‍⚕️ Clinical Portal")
    st.sidebar.write("User: Dr. Administrator")
    if st.sidebar.button("Log Out"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("🧠 Stroke Risk Engine (XGBoost)")
    
    # LOAD DATASET
    file_path = "stroke_prediction_dataset_800_rows.csv"
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        X = data.drop("stroke", axis=1)
        y = data["stroke"]
        
        # Train Model with your specific settings
        model = XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)
        model.fit(X, y)
        st.success(f"✅ AI Engine Synchronized with {len(data)} clinical records.")
    else:
        st.error(f"❌ Dataset not found! Please ensure '{file_path}' is in your GitHub.")
        st.stop()

    # GUI INPUTS
    st.markdown("---")
    st.subheader("Patient Clinical Profile")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age:", 1.0, 120.0, 50.0)
        gender = st.selectbox("Gender:", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
        hyper = st.selectbox("Hypertension:", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        heart = st.selectbox("Heart Disease:", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")

    with c2:
        glucose = st.number_input("Glucose Level:", value=100.0)
        bmi = st.number_input("BMI:", value=25.0)
        smoking = st.selectbox("Smoking Status:", [0, 1, 2], format_func=lambda x: ["Never", "Former", "Smokes"][x])
        chol = st.number_input("Cholesterol Level:", value=200.0)

    with c3:
        sys_bp = st.number_input("Systolic BP:", value=120.0)
        dia_bp = st.number_input("Diastolic BP:", value=80.0)
        act = st.selectbox("Physical Activity (0-2):", [0, 1, 2])
        alc = st.selectbox("Alcohol Intake:", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")

    # 3-TIER PREDICTION LOGIC
    if st.button("RUN FULL DIAGNOSIS", use_container_width=True):
        patient_data = pd.DataFrame([[
            age, gender, hyper, heart, glucose, bmi, smoking, chol, sys_bp, dia_bp, act, alc
        ]], columns=X.columns)

        risk_score = model.predict_proba(patient_data)[0][1]
        
        st.divider()
        if risk_score < 0.30:
            st.success(f"### Result: No Stroke detected.\n**Risk Score: {risk_score:.2%}**")
        elif 0.30 <= risk_score < 0.70:
            st.warning(f"### Result: Potential Stroke Risk.\n**Risk Score: {risk_score:.2%}**\n\n*Advice: Schedule follow-up for prevention.*")
        else:
            st.error(f"### Result: High/Immediate Stroke Risk!\n**Risk Score: {risk_score:.2%}**")

        # Download Option
        report = f"ST. MICHAEL CLINICAL REPORT\nPatient Age: {age}\nRisk Score: {risk_score:.2%}"
        st.download_button("📄 Print Summary", report, "stroke_report.txt")

# --- 4. NAVIGATION LOGIC ---
if not st.session_state.logged_in:
    login_screen()
else:
    main_app()
