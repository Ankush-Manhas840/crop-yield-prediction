import os
import threading
import time

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="centered")

MODEL_PATH = "models/model.pkl"
COLUMNS_PATH = "models/columns.pkl"
OPTIONS_PATH = "models/options.pkl"
PIPELINE_SVG_PATH = "assets/pipeline.svg"


@st.cache_resource
def load_cached_model():
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    options = joblib.load(OPTIONS_PATH)
    return model, columns, options


def train_and_cache():
    with st.status(
        "First run on this server — no cached model found. Training now, this usually takes 1-2 minutes.",
        expanded=True,
    ) as status:
        log_lines = ["$ starting up..."]
        log_box = st.empty()
        log_box.code("\n".join(log_lines), language="bash")

        df = pd.read_csv("data/crop_clean.csv")
        df = df[~((df.Crop == "Sugarcane") & (df.Yield > 200))]
        log_lines.append(f"$ loaded data/crop_clean.csv ({len(df):,} rows)")
        log_box.code("\n".join(log_lines), language="bash")

        X = pd.get_dummies(
            df[["State_Name", "Crop_Year", "Season", "Crop", "Area"]],
            columns=["State_Name", "Season", "Crop"],
        )
        y = df["Yield"]
        log_lines.append(f"$ built {X.shape[1]} one-hot features")
        log_lines.append("$ training RandomForestRegressor (100 trees)...")
        log_box.code("\n".join(log_lines), language="bash")

        result = {}

        def _fit():
            m = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            m.fit(X, y)
            result["model"] = m

        timer = st.empty()
        thread = threading.Thread(target=_fit)
        start = time.time()
        thread.start()
        while thread.is_alive():
            elapsed = time.time() - start
            timer.metric("Training elapsed", f"{elapsed:.0f}s")
            time.sleep(1)
        thread.join()
        elapsed = time.time() - start
        timer.metric("Training elapsed", f"{elapsed:.0f}s (done)")

        model = result["model"]
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

        log_lines.append(f"$ training complete in {elapsed:.0f}s, model cached for future runs")
        log_box.code("\n".join(log_lines), language="bash")
        status.update(
            label=f"Model trained in {elapsed:.0f}s and cached — this only happens once.",
            state="complete",
        )

    return model, columns, options


# Trained model isn't checked into the repo (see README). If two visitors both
# hit a container that has never trained yet, both may train concurrently and
# overwrite each other's saved files - harmless duplicate work, not a correctness bug.
if os.path.exists(MODEL_PATH):
    model, columns, options = load_cached_model()
else:
    model, columns, options = train_and_cache()

st.title("🌾 Crop Yield Predictor")
st.caption("Predicts yield (tonnes/hectare) for Rice, Wheat, Maize, and Sugarcane in India.")

with st.expander("How this model works", expanded=False):
    with open(PIPELINE_SVG_PATH, encoding="utf-8") as f:
        components.html(f.read(), height=580, scrolling=False)

st.divider()

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
