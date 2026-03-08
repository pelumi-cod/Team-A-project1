import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# 1. Page Configuration
st.set_page_config(page_title="St. Michael AI Hospital", page_icon="🏥", layout="wide")

# 2. Initialize Session State for Login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- LOGIN PAGE FUNCTION ---
def login_page():
    st.title("🏥 St. Michael AI Hospital")
    st.subheader("Clinical Decision Support System")
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.info("Authorized Personnel Only")
        username = st.text_input("Doctor ID")
        password = st.text_input("Access Key", type="password")
        
        if st.button("Authorize Access", use_container_width=True):
            if username == "admin" and password == "hospital123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid Credentials")

# --- MAIN APP FUNCTION ---
def main_app():
    st.sidebar.title("👨‍⚕️ Staff Portal")
    if st.sidebar.button("Log Out"):
        st.session_state.logged_in = False
        st.rerun()
    
    st.title("🧠 Stroke Risk Diagnostic Engine")
    
    # Internal Demo Dataset
    data = pd.DataFrame({
        'gender':[1,0,1,0,1], 'age':[60,45,70,30,80], 'hypertension':[0,1,1,0,1],
        'heart_disease':[1,0,1,0,1], 'ever_married':[1,1,1,0,1], 'work_type':[0,1,2,3,0],
        'Residence_type':[1,0,1,0,1], 'avg_glucose_level':[105,95,130,80,200],
        'bmi':[27,24,30,22,35], 'smoking_status':[2,0,1,0,2], 'stroke':[1,0,1,0,1]
    })
    
    X = data.drop('stroke', axis=1)
    y = data['stroke']
    model = GradientBoostingClassifier()
    model.fit(X, y)

    st.markdown("### Patient Vitals Entry")
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x==1 else "Female")
        age = st.number_input("Age", 1, 120, 45)
        hyp = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    with c2:
        gluc = st.number_input("Avg Glucose Level", value=100.0)
        bmi = st.number_input("BMI", value=25.0)
        smoke = st.selectbox("Smoking Status", [0,1,2], format_func=lambda x: ["Never", "Formerly", "Active"][x])

    # Corrected Indentation for Button
    if st.button("Analyze Stroke Probability", use_container_width=True):
        # We add dummy values for the columns not in the UI to match the 10 features
        inputs = [[gender, age, hyp, 0, 1, 1, 1, gluc, bmi, smoke]]
        prediction = model.predict(inputs)[0]
        
        st.divider()
        res = "HIGH RISK" if prediction == 1 else "LOW RISK"
        if prediction == 1:
            st.error(f"### ⚠️ {res}: Clinical indicators suggest liability for Stroke.")
        else:
            st.success(f"### ✅ {res}: No immediate stroke indicators detected.")

        report = f"ST. MICHAEL AI HOSPITAL REPORT\nPatient Age: {age}\nDiagnosis: {res}"
        st.download_button("📄 Download Report", report, "report.txt", "text/plain", use_container_width=True)

# --- ROUTING ---
if __name__ == "__main__":
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()
