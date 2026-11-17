import streamlit as st
import numpy as np
import pickle

# Page config 
st.set_page_config(
    page_title="Smart Irrigation System",
    page_icon="🌱",
    layout="centered"
)

# Load model & scaler 
@st.cache_resource
def load_model():
    with open("model/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# Header 
st.title("🌱 Smart Irrigation System")
st.markdown("Enter real-time sensor readings to predict whether irrigation is needed.")
st.divider()

# Input section
st.subheader("📡 Sensor Inputs")

col1, col2 = st.columns(2)

with col1:
    soil_moisture   = st.number_input("Soil Moisture",        min_value=0,   max_value=100, value=30)
    soil_humidity   = st.number_input("Soil Humidity (%)",    min_value=0,   max_value=100, value=40)

with col2:
    temperature     = st.number_input("Temperature (°C)",     min_value=0,   max_value=60,  value=28)
    air_temperature = st.number_input("Air Temperature (°C)", min_value=-10, max_value=60,  value=32)

# Engineered features (auto computed)
moisture_temp     = soil_moisture * temperature
moisture_humidity = soil_moisture * soil_humidity

st.divider()

# Show computed features
with st.expander("🔧 Engineered Features (auto-computed)"):
    st.write(f"**moisture × temperature** = {moisture_temp}")
    st.write(f"**moisture × humidity**    = {moisture_humidity}")

# Predict button
st.subheader("🤖 Prediction")

if st.button("Predict Irrigation Status", use_container_width=True, type="primary"):

    input_data = np.array([[
        soil_moisture,
        temperature,
        soil_humidity,
        air_temperature,
        moisture_temp,
        moisture_humidity
    ]])

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]

    st.divider()

    if prediction == 1:
        st.success("## 💧 Irrigation Needed — Motor ON")
        st.metric("Confidence", f"{probability[1]*100:.1f}%")
    else:
        st.info("## ✅ No Irrigation Needed — Motor OFF")
        st.metric("Confidence", f"{probability[0]*100:.1f}%")

    # Probability bar
    st.markdown("**Prediction Probability**")
    col_a, col_b = st.columns(2)
    col_a.metric("Motor OFF (0)", f"{probability[0]*100:.1f}%")
    col_b.metric("Motor ON  (1)", f"{probability[1]*100:.1f}%")

    st.progress(float(probability[1]))

# Footer
st.divider()
st.caption("Model: XGBoost | Accuracy: 73.1% | Features: Soil Moisture, Temperature, Soil Humidity, Air Temperature + 2 engineered features")
