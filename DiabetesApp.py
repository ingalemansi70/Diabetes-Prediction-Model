import streamlit as st
import pandas as pd
import joblib

model = joblib.load("diabetes_model.joblib")
scaler = joblib.load("diabetes_scaler.joblib")
columns = joblib.load("diabetes_columns.joblib")

st.title("Diabetes Prediction App")

Pregnancies = st.number_input("Pregnancies", 0, 20, 1)
Glucose = st.number_input("Glucose", 0, 200, 120)
BloodPressure = st.number_input("BloodPressure", 0, 140, 70)
SkinThickness = st.number_input("SkinThickness", 0, 100, 20)
Insulin = st.number_input("Insulin", 0, 900, 80)
BMI = st.number_input("BMI", 0.0, 70.0, 25.0)
DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", 0.0, 3.0, 0.5)
Age = st.number_input("Age", 1, 100, 30)

if st.button("Predict"):
    input_data = pd.DataFrame([[
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age
    ]], columns=columns)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("Heart Disease: Yes")
    else:
        st.success("Heart Disease: No")