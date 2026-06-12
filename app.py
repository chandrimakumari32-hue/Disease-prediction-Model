import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("disease_model.joblib")
le = joblib.load("label_encoder.joblib")

# Load dataset for dropdowns
df = pd.read_csv("healthcare_data.csv")

# ---------------- UI STYLE ---------------- #
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #E3F2FD, #BBDEFB, #90CAF9);
}
h1, h2, h3, h4, h5, h6, p, label {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #
st.markdown("""
<h1 style='text-align: center; color: white;'>
🏥 Disease Prediction System
</h1>
<h4 style='text-align: center; color: #d9d9d9;'>
AI-Powered Healthcare Prediction
</h4>
""", unsafe_allow_html=True)

st.write("Predict medical conditions using machine learning.")

st.info("""
⚠️ Disclaimer:
This application is developed for educational and research purposes only.
It is not a medical tool and should not be used for real diagnosis.
Always consult a qualified doctor for medical advice.
""")
# ---------------- INPUTS ---------------- #
age = st.number_input("Age", 1, 100)
billing = st.number_input("Billing Amount", 0.0)

admission = st.selectbox("Admission Type", ['Elective','Routine','Urgent','Emergency'])

medication = st.selectbox("Medication", sorted(df["Medication"].unique()))
test_result = st.selectbox("Test Results", ['Normal','Inconclusive','Abnormal'])

length_stay = st.number_input("Length of Stay", 1)

gender = st.selectbox("Gender", sorted(df["Gender"].unique()))
blood = st.selectbox("Blood Type", sorted(df["Blood Type"].unique()))

# ---------------- PREDICTION ---------------- #
if st.button("Predict Disease"):

    input_df = pd.DataFrame({
        "Age":[age],
        "Billing Amount":[billing],
        "Admission Type":[admission],
        "Medication":[medication],
        "Test Results":[test_result],
        "Length of Stay":[length_stay],
        "Gender":[gender],
        "Blood Type":[blood]
    })

    pred = model.predict(input_df)
    disease = le.inverse_transform(pred)

    st.markdown(f"""
<div style="
background-color:#4CAF50;
padding:20px;
border-radius:10px;
text-align:center;
font-size:28px;
font-weight:bold;
color:white;">
Predicted Disease: {disease[0]}
</div>
""", unsafe_allow_html=True)
    st.balloons()

st.markdown("---")
st.caption("© 2026 Disease Prediction System | Academic Project")