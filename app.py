import os

import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = "models/model.pkl"
COLUMNS_PATH = "models/columns.pkl"
OPTIONS_PATH = "models/options.pkl"


@st.cache_resource
def load_or_train():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        columns = joblib.load(COLUMNS_PATH)
        options = joblib.load(OPTIONS_PATH)
        return model, columns, options

    df = pd.read_csv("data/crop_clean.csv")
    df = df[~((df.Crop == "Sugarcane") & (df.Yield > 200))]

    X = pd.get_dummies(
        df[["State_Name", "Crop_Year", "Season", "Crop", "Area"]],
        columns=["State_Name", "Season", "Crop"],
    )
    y = df["Yield"]

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    columns = X.columns.tolist()
    options = {
        "states": sorted(df.State_Name.unique()),
        "crops": sorted(df.Crop.unique()),
        "seasons": sorted(df.Season.unique()),
        "years": sorted(df.Crop_Year.unique()),
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(columns, COLUMNS_PATH)
    joblib.dump(options, OPTIONS_PATH)

    return model, columns, options


model, columns, options = load_or_train()

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
