import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
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

# === Train Linear Regression Model ===
predicted_full_year_lin = []
predicted_full_year_lstm = []
for month in range(1, 13):
    month_data = df[df['Month'] == month]
    X_lin = month_data[['Year']].values
    y_lin = month_data['Demand'].values
    lin_model = LinearRegression()
    lin_model.fit(X_lin, y_lin)
    predicted_full_year_lin.append(lin_model.predict([[2025]])[0])
    
    # Extract LSTM Training Data
    X_train = np.array(X_scaled[df['Month'] == month]).reshape(-1, 1, 4)
    y_train = np.array(y_scaled[df['Month'] == month])
    
    lstm_model = Sequential([
        LSTM(128, activation='relu', return_sequences=True, input_shape=(1, 4)),
        Dropout(0.2),
        LSTM(64, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dense(1)
    ])
    
    lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    early_stopping = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1)
    lstm_model.fit(X_train, y_train, epochs=500, verbose=0, callbacks=[early_stopping, reduce_lr])
    
    input_data = np.array([[2025, np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12), month_data.iloc[-1]['Demand']]])
    input_scaled = scaler_x.transform(input_data).reshape(1, 1, 4)
    predicted_demand_scaled = lstm_model.predict(input_scaled)[0][0]
    predicted_full_year_lstm.append(scaler_y.inverse_transform([[predicted_demand_scaled]])[0][0])

real_values_2025 = [
    748, 860, 1210, 1519, 1405, 1399, 1276, 1248, 1406, 1265, 1165, 893
]

# === Comparison Graph ===
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(range(1, 13), predicted_full_year_lin, color='red', marker='o', linestyle='-', linewidth=2, label='Linear Prediction 2025')
ax2.plot(range(1, 13), predicted_full_year_lstm, color='purple', marker='s', linestyle='-', linewidth=2, label='LSTM Prediction 2025')
ax2.plot(range(1, 13), real_values_2025, color='green', marker='x', linestyle='-', linewidth=2, label='Real 2025')
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(month_names, rotation=45)
ax2.set_xlabel("Month")
ax2.set_ylabel("Maximum Demand (MW)")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)
