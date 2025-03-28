import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# === Title and Description ===
st.title("Monthly Demand Prediction for 2025")
st.write("This app predicts the monthly maximum electricity demand for the year 2025 based on past data.")

# === Define the data ===
data = {
    'Year': [2022]*12 + [2023]*12 + [2024]*12,
    'Month': list(range(1, 13)) * 3,
    'Demand': [
        595, 699, 1110, 1090, 1110, 1143, 1081, 1131, 1108, 1082, 958, 779,
        735, 852, 970, 1291, 1280, 1283, 1308, 1293, 1284, 1284, 1028, 846,
        748, 860, 1210, 1519, 1405, 1399, 1276, 1248, 1406, 1265, 1165, 893,
    ]
}
df = pd.DataFrame(data)

# === Sidebar for user selection ===
month_names = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
selected_month = st.selectbox("Select a month to predict", month_names)
month_index = month_names.index(selected_month) + 1

# === Train a linear regression model for the selected month ===
month_data = df[df['Month'] == month_index]
X = month_data['Year'].values.reshape(-1, 1)
y = month_data['Demand'].values

lin_model = LinearRegression()
lin_model.fit(X, y)
predicted_demand_lin = lin_model.predict([[2025]])[0]

# === Train an LSTM model ===
X_train = np.array(month_data['Year']).reshape(-1, 1, 1)
y_train = np.array(month_data['Demand'])

lstm_model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(1, 1)),
    LSTM(50, activation='relu'),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train, y_train, epochs=200, verbose=0)

predicted_demand_lstm = lstm_model.predict(np.array([[[2025]]]))[0][0]

# === Real values for 2025 ===
real_values_2025 = [
    748, 860, 1210, 1519, 1405, 1399, 1276, 1248, 1406, 1265, 1165, 893
]

# === Display predictions ===
st.success(f"Linear Regression Prediction for {selected_month} 2025: **{predicted_demand_lin:.2f} MW**")
st.success(f"LSTM Prediction for {selected_month} 2025: **{predicted_demand_lstm:.2f} MW**")
st.success(f"Actual Demand for {selected_month} 2025: **{real_values_2025[month_index - 1]} MW**")

# === Visualization ===
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(X, y, marker='o', color='blue', linestyle='--', label=f"{selected_month} Demand")
ax.scatter(2025, predicted_demand_lin, color='red', s=100, label='Linear Prediction (2025)')
ax.scatter(2025, predicted_demand_lstm, color='purple', s=100, marker='s', label='LSTM Prediction (2025)')
ax.scatter(2025, real_values_2025[month_index - 1], color='green', marker='x', s=100, label='Real Value (2025)')
ax.set_xlabel("Year")
ax.set_ylabel("Maximum Demand (MW)")
ax.legend()
ax.grid(True)
st.pyplot(fig)
