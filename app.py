import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/model.pkl")
columns = joblib.load("models/columns.pkl")
options = joblib.load("models/options.pkl")

st.title("Crop Yield Predictor")

state = st.selectbox("State", options["states"])
crop = st.selectbox("Crop", options["crops"])
season = st.selectbox("Season", options["seasons"])
year = st.selectbox("Year", options["years"])
area = st.number_input("Area (hectares)", min_value=1.0, value=100.0)

if st.button("Predict"):
    row = pd.DataFrame([[0] * len(columns)], columns=columns)
    row["Crop_Year"] = year
    row["Area"] = area
    row[f"State_Name_{state}"] = 1
    row[f"Season_{season}"] = 1
    row[f"Crop_{crop}"] = 1

    pred = model.predict(row)[0]
    st.success(f"Predicted yield: {pred:.2f} tonnes per hectare")
    st.info(f"Expected production: {pred * area:,.0f} tonnes")