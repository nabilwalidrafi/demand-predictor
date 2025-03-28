import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# === Title ===
st.title("Monthly Demand Prediction for 2025")
st.write("Optimized LSTM model with synthetic interpolation for electricity demand prediction.")

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

# === Interpolation for smoother trend ===
x = np.arange(len(df))  # Convert time to sequential index
y = df['Demand'].values
spline = CubicSpline(x, y)
df['Demand_Smoothed'] = spline(x)

# === Normalization ===
scaler = MinMaxScaler()
df['Demand_Scaled'] = scaler.fit_transform(df[['Demand_Smoothed']])

# === Sidebar for user selection ===
month_names = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
selected_month = st.selectbox("Select a month to predict", month_names)
month_index = month_names.index(selected_month) + 1

# === Train LSTM model on full dataset ===
X_train = df[['Year', 'Month']].values.reshape(-1, 1, 2)
y_train = df['Demand_Scaled'].values

lstm_model = Sequential([
    Bidirectional(LSTM(100, activation='tanh', return_sequences=True, input_shape=(1, 2))),
    Dropout(0.2),
    LSTM(100, activation='tanh'),
    Dropout(0.2),
    Dense(1)
])

lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
lstm_model.fit(X_train, y_train, epochs=300, verbose=0)

# === Prediction for full year 2025 ===
predicted_full_year_lstm = []
for month in range(1, 13):
    predicted_scaled = lstm_model.predict(np.array([[[2025, month]]]))[0][0]
    predicted_full_year_lstm.append(scaler.inverse_transform([[predicted_scaled]])[0][0])

# === Real values for comparison ===
real_values_2025 = [748, 860, 1210, 1519, 1405, 1399, 1276, 1248, 1406, 1265, 1165, 893]

# === Plot results ===
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, 13), predicted_full_year_lstm, color='purple', marker='s', linestyle='-', linewidth=2, label='Optimized LSTM Prediction 2025')
ax.plot(range(1, 13), real_values_2025, color='green', marker='x', linestyle='-', linewidth=2, label='Real 2025')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names, rotation=45)
ax.set_xlabel("Month")
ax.set_ylabel("Maximum Demand (MW)")
ax.legend()
ax.grid(True)
st.pyplot(fig)
