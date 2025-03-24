import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# === Title and Description ===
st.title("Monthly Demand Prediction for 2025")
st.write("This app predicts the monthly maximum electricity demand for the year 2025 based on past data using Exponential Smoothing (Holt-Winters).")

# === Define the data ===
data = {
    'Year': [2022] * 12 + [2023] * 12 + [2024] * 12,
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

# === Get corresponding month index ===
month_index = month_names.index(selected_month) + 1

# === Prepare data for Holt-Winters ===
month_data = df[df['Month'] == month_index]['Demand'].values

# === Train Holt-Winters model ===
hw_model = ExponentialSmoothing(month_data, seasonal_periods=3, trend='add', seasonal='add').fit()

# === Predict demand for 2025 ===
predicted_demand = hw_model.forecast(1)[0]

# === Real values for 2025 ===
real_values_2025 = [
    748, 860, 1210, 1519, 1405, 1399, 1276, 1248, 1406, 1265, 1165, 893
]

# === Display the prediction ===
st.success(f"Predicted Maximum Demand for {selected_month} 2025: **{predicted_demand:.2f} MW**")

# === Visualization for selected month ===
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df[df['Month'] == month_index]['Year'], df[df['Month'] == month_index]['Demand'], marker='o', color='blue', linestyle='--', label=f"{selected_month} Demand")
ax.scatter(2025, predicted_demand, color='red', s=100, label='Prediction (2025)')
ax.scatter(2025, real_values_2025[month_index - 1], color='green', marker='x', s=100, label='Real Value (2025)')
ax.set_xlabel("Year")
ax.set_ylabel("Maximum Demand (MW)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# === Full-year comparison graph using Holt-Winters ===
months_range = np.array(list(range(1, 13))).reshape(-1, 1)
predicted_full_year = []

for month in range(1, 13):
    month_data_all = df[df['Month'] == month]['Demand'].values
    hw_model_all = ExponentialSmoothing(month_data_all, seasonal_periods=3, trend='add', seasonal='add').fit()
    predicted_full_year.append(hw_model_all.forecast(1)[0])

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(months_range, predicted_full_year, color='red', marker='o', linestyle='-', linewidth=2, label='Predicted 2025')
ax2.plot(months_range, real_values_2025, color='green', marker='x', linestyle='-', linewidth=2, label='Real 2025')
ax2.set_xticks(months_range.flatten())
ax2.set_xticklabels(month_names)
ax2.set_xlabel("Month")
ax2.set_ylabel("Maximum Demand (MW)")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# === Show full dataset ===
if st.checkbox("Show full dataset"):
    st.write(df)