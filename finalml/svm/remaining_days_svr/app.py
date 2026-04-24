"""
app.py

Purpose: Streamlit web app for shelf life prediction using trained ML model.
Uses ONLY the ML model - no rule-based logic in the final prediction.
UI collects VOC / Temperature / Humidity, but the current trained model uses VOC only.
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Path to the trained ML model saved by train_remaining_days.py.
# Note: the filename is kept as logreg_model.joblib for compatibility, but the
# model itself can be a DecisionTreeClassifier (or any sklearn estimator).
MODEL_PATH = r"C:\Users\Alekhya\Desktop\4-2 project\finalml\svm (3)\svm (2)\svm\remaining_days_svr\outputs\logreg_model.joblib"


def load_model():
    """
    Load the trained ML model.
    
    Returns:
        Any: Trained sklearn estimator saved via joblib
        None: If model file not found
    """
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def predict_remaining_days(voc: float, model) -> int:
    """
    Predict remaining days using the trained ML model.
    
    Args:
        voc: VOC sensor value
        model: Trained sklearn estimator
    
    Returns:
        int: Predicted remaining days (0-10)
    """
    # Prepare input as DataFrame with proper feature name
    X = pd.DataFrame({"VOC": [float(voc)]})
    
    # Predict using the trained model (VOC-only)
    pred = int(model.predict(X)[0])
    
    # Ensure prediction is within valid range
    return int(np.clip(pred, 0, 10))


# Configure Streamlit page
st.set_page_config(
    page_title="Shelf Life Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# App header and description
st.markdown("""
# 🌾 Shelf Life Predictor
""") 

# Input section
st.markdown("### Sensor Input")

col1, col2, col3 = st.columns(3)

with col1:
    voc = st.number_input(
        "**VOC**",
        value=0.01,
        min_value=0.0,
        max_value=1.0,
        step=0.001,
        format="%.4f",
        help="Enter VOC sensor reading (typically 0.001 - 0.200)",
    )

with col2:
    temp = st.number_input(
        "**Temperature**",
        value=27.0,
        step=0.1,
    )

with col3:
    hum = st.number_input(
        "**Humidity**",
        value=71.0,
        step=0.1,
    )

# Temperature and Humidity are not used by the current model.
# They are collected for UI completeness / future model upgrades.

# Prediction button
predict_button = st.button("🔮 Predict Remaining Days", use_container_width=True)

# Prediction logic
if predict_button:
    # Load the trained model
    model = load_model()
    
    if model is not None:
        # Make prediction
        predicted_days = predict_remaining_days(voc, model)
        
        # Display result
        st.markdown("---")
        st.success(f"## Predicted Remaining Days: **{predicted_days}**")
        
        # Additional context based on prediction
        if predicted_days >= 7:
            st.info("🟢 **Fresh**: Product has good shelf life remaining.")
        elif predicted_days >= 3:
            st.warning("🟡 **Moderate**: Product should be consumed soon.")
        else:
            st.error("🔴 **Critical**: Product nears end of shelf life.")
            
    else:
        st.error("❌ Model not found!")
        st.markdown("""
        Please train the model first:
        ```bash
        python train_remaining_days.py
        ```
        """)
