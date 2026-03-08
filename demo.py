# ... (Keep everything from the previous code until the Prediction Logic) ...

    if st.button("Analyze Stroke Probability", use_container_width=True):
        inputs = [[gender, age, hyp, hd, married, work, res, gluc, bmi, smoke]]
        prediction = model.predict(inputs)[0]
        
        st.divider()
        result_text = ""
        if prediction == 1:
            result_text = "HIGH RISK: Clinical indicators suggest a high liability for Stroke."
            st.error(f"### ⚠️ {result_text}")
            st.warning("Recommendation: Immediate Cardiovascular Consultation and Lipid Panel required.")
        else:
            result_text = "LOW RISK: No immediate stroke indicators detected by AI."
            st.success(f"### ✅ {result_text}")

        # --- NEW: PRINT / DOWNLOAD SESSION ---
        report_content = f"""
        ST. MICHAEL AI HOSPITAL - CLINICAL REPORT
        ------------------------------------------
        Patient Age: {age}
        Gender: {'Male' if gender==1 else 'Female'}
        BMI: {bmi}
        Glucose Level: {gluc}
        ------------------------------------------
        DIAGNOSIS: {result_text}
        DATE: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
        ------------------------------------------
        Note: This is an AI-generated decision support summary.
        """
        
        st.download_button(
            label="📄 Download Clinical Report (for Printing)",
            data=report_content,
            file_name=f"patient_report_{pd.Timestamp.now().strftime('%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
