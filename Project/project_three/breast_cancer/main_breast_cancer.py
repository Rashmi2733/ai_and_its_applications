import streamlit as st
import pickle
import numpy as np

st.title("Breast Cancer Prediction")

# Select model
st.subheader("Select Model for Classification:")
model_choice = st.selectbox("Select Model", ["Baseline Logistic Regression Model", "Fine-tuned Logistic Regression Model"])

# Load selected model

import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_filename = "base_model_breast_cancer.pkl" if model_choice == "Baseline Logistic Regression Model" else "final_tuned_model_breast_cancer.pkl"
model_path = os.path.join(current_dir, model_filename)


# model_path = "base_model_breast_cancer.pkl" if model_choice == "Baseline Logistic Regression Model" else "final_tuned_model_breast_cancer.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Inputs
st.subheader("Input value for the given features:")

radius_mean = st.slider("Mean Radius", min_value=0.00, max_value=50.00, step=0.01)
texture_mean = st.slider("Mean Texture", min_value=0.00,max_value=50.00, step = 0.01)
smoothness_mean = st.slider("Mean Smoothness", min_value=0.00,max_value=5.00, step = 0.01) 
compactness_mean = st.slider("Mean Compactness", min_value=0.00,max_value=5.00, step = 0.01) 
concavity_mean = st.slider("Mean Concavity", min_value=0.00,max_value=5.00, step = 0.01) 
concave_points_mean = st.slider("Mean Concave Points", min_value=0.00,max_value=5.00, step = 0.01) 
symmetry_mean = st.slider("Mean Symmetry", min_value=0.00,max_value=5.00, step = 0.01) 
fractal_dimension_mean = st.slider("Mean Fractal Dimension", min_value=0.00,max_value=5.00, step = 0.01) 


# Predict
if st.button("Does the patient have  Breast Cancer?"):
    features = np.array([[radius_mean, texture_mean, smoothness_mean, compactness_mean, concavity_mean,
       concave_points_mean, symmetry_mean, fractal_dimension_mean]])
    prediction = model.predict(features)
    if int(prediction[0]) == 1:
        st.write(f"Breast Cancer Predicted: Malignant (Cancerous) ({int(prediction[0])})")
    else:
        st.write(f"Breast Cancer Predicted: Benign (Non-cancerous) ({int(prediction[0])})")
