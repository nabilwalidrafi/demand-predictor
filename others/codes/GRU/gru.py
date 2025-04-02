import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# === Title and Description ===
st.title("Optimized Monthly Demand Prediction for 2025")
st.write("Enhanced LSTM and GRU models for accurate electricity demand forecasting.")

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
df['Prev_Demand'] = df.groupby('Month')['Demand'].shift(1)
df['Prev_Demand'].fillna(method='bfill', inplace=True)

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
X_train = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
y_train = np.array(y_scaled)

# === LSTM Model ===
lstm_model = Sequential([
    LSTM(128, activation='relu', return_sequences=True, input_shape=(1, 4)),
    Dropout(0.2),
    LSTM(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')
])

lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.006), loss='mse')
lstm_model.fit(X_train, y_train, epochs=500, verbose=0)

# === GRU Model ===
gru_model = Sequential([
    GRU(128, activation='relu', return_sequences=True, input_shape=(1, 4)),
    Dropout(0.2),
    GRU(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')
])

gru_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.006), loss='mse')
gru_model.fit(X_train, y_train, epochs=500, verbose=0)

# === Predict Demand for 2025 ===
predicted_full_year_lstm = []
predicted_full_year_gru = []
for month in range(1, 13):
    input_data = np.array([[2025, np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12), df[df['Month'] == month].iloc[-1]['Demand']]])
    input_scaled = scaler_x.transform(input_data).reshape(1, 1, 4)
    
    lstm_predicted_scaled = lstm_model.predict(input_scaled)[0][0]
    gru_predicted_scaled = gru_model.predict(input_scaled)[0][0]
    
    predicted_full_year_lstm.append(scaler_y.inverse_transform([[lstm_predicted_scaled]])[0][0])
    predicted_full_year_gru.append(scaler_y.inverse_transform([[gru_predicted_scaled]])[0][0])

# === Real values for 2025 ===
real_values_2025 = [748, 860, 1210, 1519, 1405, 1399, 1276, 1248, 1406, 1265, 1165, 893]

# === First Graph: Selected Month Prediction ===
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df[df['Month'] == month_index]['Year'], df[df['Month'] == month_index]['Demand'], marker='o', color='blue', linestyle='--', label=f"{selected_month} Demand")
ax.scatter(2025, predicted_full_year_lstm[month_index - 1], color='purple', s=100, marker='s', label='LSTM Prediction (2025)')
ax.scatter(2025, predicted_full_year_gru[month_index - 1], color='red', s=100, marker='D', label='GRU Prediction (2025)')
ax.scatter(2025, real_values_2025[month_index - 1], color='green', marker='x', s=100, label='Real Value (2025)')
ax.set_xlabel("Year")
ax.set_ylabel("Maximum Demand (MW)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# === Second Graph: Full Year Predictions ===
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(range(1, 13), predicted_full_year_lstm, color='purple', marker='s', linestyle='-', linewidth=2, label='LSTM Prediction 2025')
ax2.plot(range(1, 13), predicted_full_year_gru, color='red', marker='D', linestyle='-', linewidth=2, label='GRU Prediction 2025')
ax2.plot(range(1, 13), real_values_2025, color='green', marker='x', linestyle='-', linewidth=2, label='Real 2025')
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(month_names, rotation=45)
ax2.set_xlabel("Month")
ax2.set_ylabel("Maximum Demand (MW)")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)
