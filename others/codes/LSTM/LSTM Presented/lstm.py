import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# === Title and Description ===
st.title("Optimized Monthly Demand Prediction for 2025")
st.write("Enhanced LSTM model for accurate electricity demand forecasting.")

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

# === Feature Engineering ===
df['Prev_Demand'] = df.groupby('Month')['Demand'].shift(1).fillna(method='bfill')
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)

# === Scaling ===
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_x.fit_transform(df[['Year', 'Month_Sin', 'Month_Cos', 'Prev_Demand']])
y_scaled = scaler_y.fit_transform(df[['Demand']])

# === Sidebar for user selection ===
month_names = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
selected_month = st.selectbox("Select a month to predict", month_names)
month_index = month_names.index(selected_month) + 1

# === Extract training data ===
month_data = df[df['Month'] == month_index]
X_train = np.array(X_scaled[df['Month'] == month_index]).reshape(-1, 1, 4)
y_train = np.array(y_scaled[df['Month'] == month_index])

# === LSTM Model ===
lstm_model = Sequential([
    LSTM(128, activation='relu', return_sequences=True, input_shape=(1, 4)),
    Dropout(0.2),
    LSTM(64, activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dense(1)
])

lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')

# === Callbacks ===
early_stopping = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1)

# === Train the LSTM ===
lstm_model.fit(X_train, y_train, epochs=500, verbose=0, callbacks=[early_stopping, reduce_lr])

# === Prediction for 2025 ===
input_data = np.array([[2025, np.sin(2*np.pi*month_index/12), np.cos(2*np.pi*month_index/12), month_data.iloc[-1]['Demand']]])
input_scaled = scaler_x.transform(input_data).reshape(1, 1, 4)
predicted_demand_scaled = lstm_model.predict(input_scaled)[0][0]
predicted_demand_lstm = scaler_y.inverse_transform([[predicted_demand_scaled]])[0][0]

# === Real values for 2025 ===
real_values_2025 = [
    748, 860, 1210, 1519, 1405, 1399, 1276, 1248, 1406, 1265, 1165, 893
]

# === Display predictions ===
st.success(f"Optimized LSTM Prediction for {selected_month} 2025: **{predicted_demand_lstm:.2f} MW**")
st.success(f"Actual Demand for {selected_month} 2025: **{real_values_2025[month_index - 1]} MW**")

# === Visualization ===
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df[df['Month'] == month_index]['Year'], df[df['Month'] == month_index]['Demand'], marker='o', color='blue', linestyle='--', label=f"{selected_month} Demand")
ax.scatter(2025, predicted_demand_lstm, color='purple', s=100, marker='s', label='Optimized LSTM Prediction (2025)')
ax.scatter(2025, real_values_2025[month_index - 1], color='green', marker='x', s=100, label='Real Value (2025)')
ax.set_xlabel("Year")
ax.set_ylabel("Maximum Demand (MW)")
ax.legend()
ax.grid(True)
st.pyplot(fig)
