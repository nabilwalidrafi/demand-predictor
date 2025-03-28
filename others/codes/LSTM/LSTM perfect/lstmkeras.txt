import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import keras_tuner as kt

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
X_train = np.array(X_scaled).reshape(-1, 1, 4)
y_train = np.array(y_scaled)

# === Hyperparameter Tuning ===
def build_model(hp):
    model = Sequential()
    model.add(LSTM(hp.Int('units_1', 32, 128, step=32), activation='relu', return_sequences=True, input_shape=(1, 4)))
    model.add(Dropout(hp.Float('dropout_1', 0.1, 0.5, step=0.1)))
    
    if hp.Boolean('second_layer'):
        model.add(LSTM(hp.Int('units_2', 32, 128, step=32), activation='relu', return_sequences=True))
        model.add(Dropout(hp.Float('dropout_2', 0.1, 0.5, step=0.1)))
    
    model.add(LSTM(hp.Int('units_3', 32, 128, step=32), activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [0.01, 0.001, 0.0001])), loss='mse')
    return model

tuner = kt.RandomSearch(build_model, objective='val_loss', max_trials=10, executions_per_trial=1, directory='tuner_results', project_name='lstm_tuning')

tuner.search(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)

# === Train Best Model ===
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

best_model.fit(X_train, y_train, epochs=500, verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=20, restore_best_weights=True), ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1)])

# === Predict Demand for 2025 ===
predicted_full_year_lstm = []
for month in range(1, 13):
    input_data = np.array([[2025, np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12), df[df['Month'] == month].iloc[-1]['Demand']]])
    input_scaled = scaler_x.transform(input_data).reshape(1, 1, 4)
    predicted_scaled = best_model.predict(input_scaled)[0][0]
    predicted_full_year_lstm.append(scaler_y.inverse_transform([[predicted_scaled]])[0][0])

# === Real values for 2025 ===
real_values_2025 = [748, 860, 1210, 1519, 1405, 1399, 1276, 1248, 1406, 1265, 1165, 893]

# === Visualization ===
st.title("Optimized Monthly Demand Prediction for 2025")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, 13), predicted_full_year_lstm, color='purple', marker='s', linestyle='-', linewidth=2, label='Optimized LSTM Prediction 2025')
ax.plot(range(1, 13), real_values_2025, color='green', marker='x', linestyle='-', linewidth=2, label='Real 2025')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=45)
ax.set_xlabel("Month")
ax.set_ylabel("Maximum Demand (MW)")
ax.legend()
ax.grid(True)
st.pyplot(fig)
