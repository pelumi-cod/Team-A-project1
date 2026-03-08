import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 1. Page Configuration
st.set_page_config(page_title="St. Michael AI Hospital", page_icon="🏥", layout="wide")

# 2. Session State for Login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- LOGIN SCREEN ---
def login_page():
    st.title("🏥 St. Michael AI Hospital")
    st.subheader("Cloud-Based Clinical Decision Support System")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.info("Medical Staff Authentication Required")
        username = st.text_input("Staff ID (Username)")
        password = st.text_input("Access Key (Password)", type="password")
        if st.button("Authorize Access", use_container_width=True):
            if username == "admin" and password == "hospital123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Access Denied: Invalid Credentials")

# --- CLINICAL APP ---
def main_app():
    st.sidebar.title("👨‍⚕️ Staff Portal")
    if st.sidebar.button("Log Out"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("🧠 Stroke Risk Engine (XGBoost Integration)")
    st.write("Enter the complete clinical profile to execute the diagnostic scan.")

    # 3. Full 10-Feature Training Data (Your original structure)
    data = pd.DataFrame({
        'gender':[1,0,1,0,1], 'age':[60,45,70,30,80], 'hypertension':[0,1,1,0,1],
        'heart_disease':[1,0,1,0,1], 'ever_married':[1,1,1,0,1], 'work_type':[0,1,2,3,0],
        'Residence_type':[1,0,1,0,1], 'avg_glucose_level':[105,95,130,80,200],
        'bmi':[27,24,30,22,35], 'smoking_status':[2,0,1,0,2], 'stroke':[1,0,1,0,1]
    })
    
    X = data.drop('stroke', axis=1)
    y = data['stroke']
    
    # Using XGBoost as requested in your original project specs
    model = xgb.XGBClassifier()
    model.fit(X, y)

    # 4. Clinical Input Panel
    st.markdown("### Patient Vitals Entry")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        gender = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
        age = st.number_input("Age", 1, 120, 45)
        hyp = st.selectbox("Hypertension", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
        hd = st.selectbox("Heart Disease", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
    
    with c2:
        married = st.selectbox("Ever Married", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
        work = st.selectbox("Work Type", [0, 1, 2, 3, 4], format_func=lambda x: ["Private", "Self-employed", "Govt", "Children", "Never worked"][x])
        res = st.selectbox("Residence Type", [1, 0], format_func=lambda x: "Urban" if x==1 else "Rural")
    
    with c3:
        gluc = st.number_input("Avg Glucose Level", value=100.0)
        bmi = st.number_input("BMI", value=25.0)
        smoke = st.selectbox("Smoking Status", [2, 0, 1, 3], format_func=lambda x: ["Smokes", "Never Smoked", "Formerly Smoked", "Unknown"][x])

    # 5. Diagnostic Output
    if st.button("RUN DIAGNOSTIC SCAN", use_container_width=True):
        patient_features = [[gender, age, hyp, hd, married, work, res, gluc, bmi, smoke]]
        prediction = model.predict(patient_features)[0]
        
        st.divider()
        diag = "HIGH RISK" if prediction == 1 else "LOW RISK"
        
        if prediction == 1:
            st.error(f"### ⚠️ AI DIAGNOSIS: {diag}")
            st.write("Patient shows high liability for Stroke. Immediate clinical intervention advised.")
        else:
            st.success(f"### ✅ AI DIAGNOSIS: {diag}")
            st.write("Patient clinical metrics are within safe predictive ranges.")

        # --- PRINT SESSION ---
        report = f"ST. MICHAEL AI HOSPITAL - CLINICAL SUMMARY\n" + "-"*40 + \
                 f"\nAge: {age}\nGlucose: {gluc}\nBMI: {bmi}\nDiagnosis: {diag}\n" + "-"*40
        st.download_button("📄 Download/Print Clinical Report", report, "stroke_report.txt", use_container_width=True)

# --- RUNTIME ---
if __name__ == "__main__":
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()
