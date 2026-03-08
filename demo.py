import streamlit as st
import pandas as pd
import xgboost as xgb
import os
from datetime import datetime
import openpyxl 

# 1. Page Configuration
st.set_page_config(page_title="TEAM A HOSPITAL", page_icon="🏥", layout="wide")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- LOGIN SCREEN ---
def login_screen():
    st.title("🏥 TEAM A HOSPITAL")
    st.subheader("Clinical Diagnostic & Record Management System")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        user = st.text_input("Doctor ID")
        pw = st.text_input("Access Key", type="password")
        if st.button("Authorize Access", use_container_width=True):
            if user == "admin" and pw == "hospital123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid Credentials")

# --- DATABASE ENGINE ---
def save_to_excel(new_data):
    db_file = "patient_records.xlsx"
    if not os.path.exists(db_file):
        df = pd.DataFrame(columns=new_data.keys())
        df.to_excel(db_file, index=False)
    
    existing_df = pd.read_excel(db_file)
    updated_df = pd.concat([existing_df, pd.DataFrame([new_data])], ignore_index=True)
    updated_df.to_excel(db_file, index=False)

# --- MAIN CLINICAL PORTAL ---
def main_app():
    st.sidebar.title("👨‍⚕️ Doctor's Workspace")
    if st.sidebar.button("Log Out"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("🧠❤️ Clinical Risk & Patient Record System")
    
    # Load Training Data
    file_path = "stroke_prediction_dataset_800_rows.csv"
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        X_s = data.drop("stroke", axis=1)
        y_s = data["stroke"]
        model_s = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
        model_s.fit(X_s, y_s)
        
        X_h = data.drop("heart_disease", axis=1)
        y_h = data["heart_disease"]
        model_h = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
        model_h.fit(X_h, y_h)
    else:
        st.error("❌ Training Dataset missing!")
        st.stop()

    # INPUT PANEL
    st.markdown("### 📋 New Patient Entry")
    patient_name = st.text_input("Patient Full Name / ID:") 
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

    if st.button("RUN DIAGNOSIS & SAVE RECORD", use_container_width=True):
        # AI Logic
        p_stroke = pd.DataFrame([[age, gender, hyper, 0, glucose, bmi, smoke, chol, sys_bp, dia_bp, 1, 0]], columns=X_s.columns)
        s_score = float(model_stroke_prob := model_s.predict_proba(p_stroke)[0][1])
        
        p_heart = pd.DataFrame([[age, gender, hyper, glucose, bmi, smoke, chol, sys_bp, dia_bp, 1, 0, (1 if s_score > 0.5 else 0)]], columns=X_h.columns)
        h_score = float(model_h.predict_proba(p_heart)[0][1])

        # Status Assignment
        s_status = "High Risk" if s_score > 0.7 else "Likely to have soon" if s_score > 0.3 else "Low Risk"
        h_status = "High Risk" if h_score > 0.7 else "Likely to have soon" if h_score > 0.3 else "Low Risk"

        # DISPLAY RESULTS
        st.divider()
        res1, res2 = st.columns(2)
        with res1:
            st.markdown(f"### 🧠 Stroke: {s_status}")
            if "Likely" in s_status: st.info("⚠️ Recommendation: The person should go see a doctor for precautionary measures.")
        with res2:
            st.markdown(f"### ❤️ Heart: {h_status}")
            if "Likely" in h_status: st.info("⚠️ Recommendation: The person should go see a doctor for precautionary measures.")

        # --- SAVE TO EXCEL ---
        record = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Patient Name": patient_name if patient_name else "Unknown",
            "Age": age,
            "Stroke Assessment": s_status,
            "Heart Assessment": h_status,
            "Glucose": glucose,
            "BMI": bmi
        }
        save_to_excel(record)
        st.success(f"💾 File update: Record for '{patient_name}' has been securely archived.")

    # --- THE ARCHIVE SECTION (Visible on Page) ---
    st.divider()
    st.subheader("📂 Hospital Patient Archives")
    
    if os.path.exists("patient_records.xlsx"):
        history_df = pd.read_excel("patient_records.xlsx")
        
        # Display the table on the page
        st.dataframe(history_df.sort_index(ascending=False), use_container_width=True)
        
        # Download button for necessity
        with open("patient_records.xlsx", "rb") as f:
            st.download_button(
                label="📥 Download Full Database (Excel)",
                data=f,
                file_name=f"Hospital_Database_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("No records found. The database will appear here once the first diagnosis is saved.")

# --- RUNTIME ---
if not st.session_state.logged_in:
    login_screen()
else:
    main_app()

