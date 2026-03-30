# app.py
import streamlit as st
import pandas as pd
import torch
import joblib
import os

# Import your models
from models import bnn_model
from models import lstm_model
from models import transformer_model
from models import xgboost_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Load dataset and encoders
# ------------------------------
scaler_X = joblib.load("models/scaler_X.pkl")
scaler_y = joblib.load("models/scaler_y.pkl")
state_to_idx = joblib.load("models/state_to_idx.pkl")

feature_cols = [
    "ppt (mm)_AVG", "tmax (degrees C)_AVG", "tmean (degrees C)_AVG", "tmin (degrees C)_AVG",
    "ssm_AVG", "susm_AVG",
    "EVI_AVG", "GCI_AVG", "NDWI_AVG", "NDVI_AVG",
    "AWC", "SOM", "CEC", "state_idx",
    "HIST_AVG_YIELD"
]

state_names = [
    "ALABAMA","ARKANSAS","CALIFORNIA","COLORADO","DELAWARE","GEORGIA","ILLINOIS","INDIANA",
    "IOWA","KANSAS","KENTUCKY","LOUISIANA","MARYLAND","MICHIGAN","MINNESOTA","MISSISSIPPI",
    "MISSOURI","MONTANA","NEBRASKA","NEW JERSEY","NEW MEXICO","NEW YORK","NORTH CAROLINA",
    "NORTH DAKOTA","OHIO","OKLAHOMA","PENNSYLVANIA","SOUTH CAROLINA","SOUTH DAKOTA",
    "TENNESSEE","TEXAS","VIRGINIA","WEST VIRGINIA","WISCONSIN"
]

st.set_page_config(page_title="Corn Yield Prediction", layout="wide")
st.title("🌽 Corn Yield Prediction App")

# ------------------------------
# Sidebar navigation
# ------------------------------
page = st.sidebar.selectbox("Choose a page", ["Predict Yield", "Model Comparison"])

if page == "Predict Yield":
    st.header("Enter Feature Values")
    with st.form("input_form"):
        # State dropdown
        state_name = st.selectbox("Select State", state_names)
        state_idx = state_to_idx[state_name]

        # Feature inputs
        ppt = st.number_input("ppt (mm)_AVG", value=100.0)
        tmax = st.number_input("tmax (°C)_AVG", value=30.0)
        tmean = st.number_input("tmean (°C)_AVG", value=25.0)
        tmin = st.number_input("tmin (°C)_AVG", value=20.0)
        ssm = st.number_input("ssm_AVG", value=0.5)
        susm = st.number_input("susm_AVG", value=0.5)
        evi = st.number_input("EVI_AVG", value=0.2)
        gci = st.number_input("GCI_AVG", value=0.3)
        ndwi = st.number_input("NDWI_AVG", value=0.1)
        ndvi = st.number_input("NDVI_AVG", value=0.4)
        awc = st.number_input("AWC", value=0.2)
        som = st.number_input("SOM", value=0.05)
        cec = st.number_input("CEC", value=10.0)
        hist_yield = st.number_input("HIST_AVG_YIELD", value=100.0)

        model_option = st.selectbox("Select Model", ["All models", "BNN", "LSTM", "Transformer", "XGBoost"])

        submitted = st.form_submit_button("Predict Yield")

    if submitted:
        # Create dataframe
        input_df = pd.DataFrame([[
            ppt, tmax, tmean, tmin, ssm, susm,
            evi, gci, ndwi, ndvi,
            awc, som, cec, state_idx, hist_yield
        ]], columns=feature_cols)

        X_scaled = scaler_X.transform(input_df[feature_cols])
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

        results = {}

        if model_option == "All models" or model_option == "BNN":
            bnn_pred_scaled = bnn_model.predict(X_tensor, device)
            results["BNN"] = scaler_y.inverse_transform(bnn_pred_scaled).flatten()[0]

        if model_option == "All models" or model_option == "LSTM":
            lstm_pred_scaled = lstm_model.predict(X_tensor, device)
            results["LSTM"] = scaler_y.inverse_transform(lstm_pred_scaled).flatten()[0]

        if model_option == "All models" or model_option == "Transformer":
            transformer_pred_scaled = transformer_model.predict(X_tensor, device)
            results["Transformer"] = scaler_y.inverse_transform(transformer_pred_scaled).flatten()[0]

        if model_option == "All models" or model_option == "XGBoost":
            xgb_pred_scaled = xgboost_model.predict(X_scaled)
            results["XGBoost"] = scaler_y.inverse_transform(xgb_pred_scaled.reshape(-1,1)).flatten()[0]

        st.subheader("✅ Predicted Yield")
        for model_name, pred in results.items():
            st.write(f"**{model_name}:** {pred:.2f}")

# ------------------------------
# Model Comparison Page
# ------------------------------
elif page == "Model Comparison":
    st.header("📊 Model Comparison & Analysis")
    plots_folder = "plots"
    plot_files = [
        "correlation_matrix.png",
        "MSE_comparison.png",
        "R2_comparison.png",
        "shap_analysis.png",
        "train_mse.png",
        "train_r2.png"
    ]
    custom_titles = {
        "correlation_matrix.png": "Correlation Matrix",
        "MSE_comparison.png": "MSE Comparison",
        "R2_comparison.png": "R2 Comparison",
        "shap_analysis.png": "SHAP Analysis",
        "train_mse.png": "Train MSE",
        "train_r2.png": "Train R2"
    }
    for plot_file in plot_files:
        plot_path = os.path.join(plots_folder, plot_file)
        if os.path.exists(plot_path):
            plot_title = custom_titles.get(plot_file, plot_file.replace("_", " ").replace(".png", ""))
            st.markdown(f"<h4 style='text-align: center'>{plot_title}</h4>", unsafe_allow_html=True)
            st.image(plot_path, caption=plot_title, width="content")
        else:
            st.warning(f"{plot_file} not found in {plots_folder}")