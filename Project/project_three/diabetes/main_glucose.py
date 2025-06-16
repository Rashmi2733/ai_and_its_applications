import streamlit as st
import pickle
import numpy as np

st.title("Glucose Prediction:")

# Select model
st.subheader("Select Model for Classification:")
model_choice = st.selectbox("Select Model", ["Baseline Random Forest Model", "Fine-tuned Random Forest Model"])

# Load selected model

import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_filename = "base_model_glucose.pkl" if model_choice == "Baseline Random Forest Model" else "final_tuned_model_glucose.pkl"
model_path = os.path.join(current_dir, model_filename)

with open(model_path, "rb") as f:
    model = pickle.load(f)

st.subheader("Input value for the given features:")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
# glucose = st.number_input("Glucose Level", min_value=0, max_value=200, step=1)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, step=1)
# skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, step=1)
insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, step=1)
#micro units of insulin per milliliter of blood
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
dpf = st.slider("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01)
age = st.number_input("Age", min_value=1, max_value=100, step=1)

# Prediction
if st.button("Predict Glucose Levels:"):
    features = np.array([[pregnancies, blood_pressure,  insulin, bmi, dpf, age]])
    prediction = model.predict(features)

    st.write(f"Glucose level: {int(prediction[0])}")

    # if prediction[0] == 1:
    #     st.error("The model predicts that the person is likely to have diabetes.")
    # else:
    #     st.success("The model predicts that the person is not likely to have diabetes.")
