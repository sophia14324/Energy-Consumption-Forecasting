import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

model = tf.keras.models.load_model("lstm_household_power.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Household Power Consumption Predictor")

st.write("Enter the following inputs (from the prior time step) to predict `Global_active_power`:")

input_global_active_power = st.number_input("Global_active_power(t-1):", min_value=0.0, step=0.01)
input_global_reactive_power = st.number_input("Global_reactive_power(t-1):", min_value=0.0, step=0.01)
input_voltage = st.number_input("Voltage(t-1):", min_value=0.0, step=0.01)
input_global_intensity = st.number_input("Global_intensity(t-1):", min_value=0.0, step=0.01)
input_sub_metering_1 = st.number_input("Sub_metering_1(t-1):", min_value=0.0, step=0.01)
input_sub_metering_2 = st.number_input("Sub_metering_2(t-1):", min_value=0.0, step=0.01)
input_sub_metering_3 = st.number_input("Sub_metering_3(t-1):", min_value=0.0, step=0.01)

if st.button("Predict"):
    input_features = np.array([
        [
            input_global_active_power,
            input_global_reactive_power,
            input_voltage,
            input_global_intensity,
            input_sub_metering_1,
            input_sub_metering_2,
            input_sub_metering_3,
        ]
    ])

    scaled_input = scaler.transform(input_features)

    reshaped_input = np.expand_dims(scaled_input, axis=1)  # shape: (1, 1, 7)

    prediction = model.predict(reshaped_input)

    scaled_prediction = np.concatenate((prediction, scaled_input[:, 1:]), axis=1)  # Append other features
    inv_prediction = scaler.inverse_transform(scaled_prediction)[:, 0]

    st.write(f"Predicted Global_active_power: {inv_prediction[0]:.2f}")
