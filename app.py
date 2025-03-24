import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# === Data Setup ===
data = {
    'Year': [2022]*12 + [2023]*12 + [2024]*12 + [2025]*12,
    'Month': list(range(1, 13)) * 4,
    'Demand': [
        595, 699, 1110, 1090, 1110, 1143, 1081, 1131, 1108, 1082, 958, 779,
        735, 852, 970, 1291, 1280, 1283, 1308, 1293, 1284, 1284, 1028, 846,
        748, 860, 1210, 1519, 1405, 1399, 1276, 1248, 1406, 1265, 1165, 893,
        748, 860, 1210, 1519, 1405, 1399, 1276, 1248, 1406, 1265, 1165, 893
    ]
}
df = pd.DataFrame(data)

# === Train/Test Split ===
train_df = df[df['Year'] < 2025]
X_train = train_df[['Year', 'Month']]
y_train = train_df['Demand']

# === Define models ===
models = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression (Degree 2)": PolynomialFeatures(degree=2),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
}

# === Predictions Storage ===
predictions = {}

# === Train and Predict ===
for name, model in models.items():
    if name == "Polynomial Regression (Degree 2)":
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X_train)
        poly_model = LinearRegression()
        poly_model.fit(X_poly, y_train)
        X_test_poly = poly_features.transform(df[df['Year'] == 2025][['Year', 'Month']])
        predictions[name] = poly_model.predict(X_test_poly)
    else:
        model.fit(X_train, y_train)
        predictions[name] = model.predict(df[df['Year'] == 2025][['Year', 'Month']])

# === LSTM Model Setup ===
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(1, 2)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

# Prepare data for LSTM
X_lstm = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
y_lstm = np.array(y_train)

lstm_model.fit(X_lstm, y_lstm, epochs=100, verbose=0)
X_test_lstm = np.array(df[df['Year'] == 2025][['Year', 'Month']]).reshape((12, 1, 2))
predictions['LSTM'] = lstm_model.predict(X_test_lstm).flatten()

# === Prophet / Exponential Smoothing ===
hw_model = ExponentialSmoothing(y_train, seasonal='add', seasonal_periods=12).fit()
predictions['Holt-Winters'] = hw_model.forecast(12)

# === Performance Metrics ===
real_values = df[df['Year'] == 2025]['Demand'].values
st.write("### Performance Metrics")

for name, pred in predictions.items():
    mae = mean_absolute_error(real_values, pred)
    rmse = np.sqrt(mean_squared_error(real_values, pred))
    st.write(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# === Visualization ===
fig, ax = plt.subplots(figsize=(10, 6))
months = list(range(1, 13))

ax.plot(months, real_values, color='green', marker='o', linewidth=2, label='Real Demand')

for name, pred in predictions.items():
    ax.plot(months, pred, linestyle='--', marker='o', label=f'Predicted ({name})')

ax.set_xlabel("Month")
ax.set_ylabel("Maximum Demand (MW)")
ax.set_title("Real vs Predicted Demand for 2025")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# === Show full dataset ===
if st.checkbox("Show full dataset"):
    st.write(df)
