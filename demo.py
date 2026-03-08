# Ensure this block is inside the main_app() function!
    if st.button("Analyze Stroke Probability", use_container_width=True):
        inputs = [[gender, age, hyp, hd, married, work, res, gluc, bmi, smoke]]
        prediction = model.predict(inputs)[0]
        
        st.divider()
        result_text = ""
        if prediction == 1:
            result_text = "HIGH RISK: Clinical indicators suggest a high liability for Stroke."
            st.error(f"### ⚠️ {result_text}")
        else:
            result_text = "LOW RISK: No immediate stroke indicators detected by AI."
            st.success(f"### ✅ {result_text}")

        # The report section must also be indented correctly
        report_content = f"ST. MICHAEL AI HOSPITAL REPORT\nDiagnosis: {result_text}"
        
        st.download_button(
            label="📄 Download Clinical Report",
            data=report_content,
            file_name="report.txt",
            mime="text/plain",
            use_container_width=True
        )
