import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# 1. Page Config
st.set_page_config(page_title="Stroke Prediction AI", page_icon="🧠")

# 2. Create Dataset (Small demo set)
data = pd.DataFrame({
    'gender':[1,0,1],
    'age':[60,45,70],
    'hypertension':[0,1,1],
    'heart_disease':[1,0,1],
    'ever_married':[1,1,1],
    'work_type':[0,1,2],
    'Residence_type':[1,0,1],
    'avg_glucose_level':[105.5,95.2,130.6],
    'bmi':[27.8,24.1,30.5],
    'smoking_status':[2,0,1],
    'stroke':[1,0,1]
})

# 3. Train Model
X = data.drop('stroke', axis=1)
y = data['stroke']
model = GradientBoostingClassifier()
model.fit(X, y)

# 4. Streamlit User Interface
st.title("🧠 Stroke Prediction AI")
st.write("Enter patient details below to predict stroke liability.")

# Replacing input() with Streamlit widgets
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x==1 else "Female")
    age = st.number_input("Age", 1, 120, 45)
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    ever_married = st.selectbox("Ever Married", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

with col2:
    work_type = st.selectbox("Work Type", [0, 1, 2, 3, 4], 
                            format_func=lambda x: ["Private", "Self-employed", "Govt job", "Children", "Never worked"][x])
    residence = st.selectbox("Residence Type", [0, 1], format_func=lambda x: "Urban" if x==1 else "Rural")
    glucose = st.number_input("Average Glucose Level", value=100.0)
    bmi = st.number_input("BMI", value=25.0)
    smoking = st.selectbox("Smoking Status", [0, 1, 2, 3], 
                          format_func=lambda x: ["Never smoked", "Formerly smoked", "Smokes", "Unknown"][x])

# 5. Prediction Logic
if st.button("Predict Stroke Risk", use_container_width=True):
    patient_data = {
        'gender': gender, 'age': age, 'hypertension': hypertension,
        'heart_disease': heart_disease, 'ever_married': ever_married,
        'work_type': work_type, 'Residence_type': residence,
        'avg_glucose_level': glucose, 'bmi': bmi, 'smoking_status': smoking
    }
    
    df = pd.DataFrame([patient_data])
    prediction = model.predict(df)[0]
    
    if prediction == 1:
        st.error("⚠️ Prediction: This Patient is liable to have a Stroke")
    else:
        st.success("✅ Prediction: Low Risk / No Stroke Detected")