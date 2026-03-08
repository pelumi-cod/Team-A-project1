import streamlit as st
import pandas as pd
import xgboost as xgb
import os

# 1. Page Configuration
st.set_page_config(page_title="TEAM A HOSPITAL", page_icon="🏥", layout="wide")

# 2. Session State for Login & History
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Time', 'Age', 'Diagnosis'])

# --- LOGIN SCREEN ---
def login_page():
    st.title("🏥 TEAM A HOSPITAL")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.subheader("Clinical Portal Login")
        user = st.text_input("Doctor ID")
        pw = st.text_input("Access Key", type="password")
        if st.button("Authorize", use_container_width=True):
            if user == "admin" and pw == "hospital123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid Credentials")

# --- MAIN APP ---
def main_app():
    st.sidebar.title("👨‍⚕️ Staff Menu")
    if st.sidebar.button("Log Out"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("🧠 Stroke Diagnostic Engine (XGBoost + 800 Records)")
    
    # 3. Load Your 800-Row Dataset
    path = "stroke_prediction_dataset_800_rows.csv"
    
    if os.path.exists(path):
        df_train = pd.read_csv(path)
        # Separate features (X) and target (y)
        X = df_train.drop('stroke', axis=1)
        y = df_train['stroke']
        
        # Train XGBoost on your 800 rows
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X, y)
        st.success(f"✅ AI Engine Synchronized: {len(df_train)} clinical records loaded.")
    else:
        st.error(f"❌ Error: '{path}' not found in GitHub. Please upload the CSV.")
        st.stop()

    # 4. Input Panel (Matching your 12 CSV features exactly)
    st.markdown("### Patient Clinical Vitals")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        age = st.number_input("Age", 1, 120, 50)
        gender = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
        hyp = st.selectbox("Hypertension", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
        hd = st.selectbox("Heart Disease", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
    
    with c2:
        glucose = st.number_input("Avg Glucose Level", value=110.0)
        bmi = st.number_input("BMI", value=28.0)
        smoke = st.selectbox("Smoking Status", [0, 1, 2], format_func=lambda x: ["Never", "Formerly", "Active"][x])
        chol = st.number_input("Cholesterol", value=200.0)

    with c3:
        sys_bp = st.number_input("Systolic BP", value=120.0)
        dia_bp = st.number_input("Diastolic BP", value=80.0)
        phys = st.selectbox("Physical Activity", [0, 1, 2], format_func=lambda x: ["Low", "Moderate", "High"][x])
        alcohol = st.selectbox("Alcohol Intake", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

    # 5. Diagnostic Output
    if st.button("EXECUTE XGBOOST ANALYSIS", use_container_width=True):
        # The feature list MUST be in the exact order of the CSV columns
        patient_data = pd.DataFrame([[
            age, gender, hyp, hd, glucose, bmi, smoke, chol, sys_bp, dia_bp, phys, alcohol
        ]], columns=X.columns)
        
        prediction = model.predict(patient_data)[0]
        
        st.divider()
        diag = "HIGH STROKE RISK" if prediction == 1 else "NORMAL / LOW RISK"
        
        if prediction == 1:
            st.error(f"### ⚠️ AI DIAGNOSIS: {diag}")
            st.warning("Note: Clinical indicators suggest a high liability for stroke.")
        else:
            st.success(f"### ✅ AI DIAGNOSIS: {diag}")
            st.info("Note: Clinical metrics are within standard safety thresholds.")

        # Save to session 'database' table
        new_row = {'Time': pd.Timestamp.now().strftime('%H:%M'), 'Age': age, 'Diagnosis': diag}
        st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([new_row])], ignore_index=True)
        
        # Print Session (Download)
        report = f"ST. MICHAEL CLINICAL SUMMARY\nAge: {age}\nDiagnosis: {diag}\nEngine: XGBoost"
        st.download_button("📄 Print Clinical Report", report, "stroke_report.txt", use_container_width=True)

    # 6. Session History
    st.divider()
    st.subheader("📊 Session Patient Logs")
    st.dataframe(st.session_state.history, use_container_width=True)

# --- RUNTIME ---
if __name__ == "__main__":
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()
