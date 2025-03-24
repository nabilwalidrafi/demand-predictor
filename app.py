import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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

# === Get corresponding month index ===
month_index = month_names.index(selected_month) + 1

# === Train a separate model for the selected month ===
month_data = df[df['Month'] == month_index]
X = month_data['Year'].values.reshape(-1, 1)
y = month_data['Demand'].values

model = LinearRegression()
model.fit(X, y)

# === Holt-Winters Exponential Smoothing Model ===
hw_model = ExponentialSmoothing(y, seasonal='add', seasonal_periods=12).fit()
hw_predicted_demand = hw_model.forecast(12)

# === Predict demand for 2025 using linear regression and Holt-Winters ===
predicted_demand_lr = model.predict([[2025]])[0]
predicted_demand_hw = hw_predicted_demand[month_index - 1]

# === Real values for 2025 ===
real_values_2025 = [
    748, 860, 1210, 1519, 1405, 1399, 1276, 1248, 1406, 1265, 1165, 893
]

# === Display the prediction ===
st.success(f"Predicted Maximum Demand for {selected_month} 2025 (Linear Regression): **{predicted_demand_lr:.2f} MW**")
st.success(f"Predicted Maximum Demand for {selected_month} 2025 (Holt-Winters): **{predicted_demand_hw:.2f} MW**")

# === Visualization for selected month ===
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(X, y, marker='o', color='blue', linestyle='--', label=f"{selected_month} Demand")
ax.scatter(2025, predicted_demand_lr, color='red', s=100, label='Prediction (2025 - LR)')
ax.scatter(2025, predicted_demand_hw, color='orange', s=100, label='Prediction (2025 - Holt-Winters)')
ax.scatter(2025, real_values_2025[month_index - 1], color='green', marker='x', s=100, label='Real Value (2025)')
ax.set_xlabel("Year")
ax.set_ylabel("Maximum Demand (MW)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# === Full-year comparison graph ===
months_range = np.array(list(range(1, 13))).reshape(-1, 1)
predicted_full_year_lr = []
predicted_full_year_hw = []

for month in range(1, 13):
    month_data = df[df['Month'] == month]
    X_month = month_data['Year'].values.reshape(-1, 1)
    y_month = month_data['Demand'].values
    model_month = LinearRegression()
    model_month.fit(X_month, y_month)
    predicted_full_year_lr.append(model_month.predict([[2025]])[0])

    # Holt-Winters Forecast for Full Year
    predicted_full_year_hw.append(hw_model.forecast(12)[month - 1])

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(months_range, predicted_full_year_lr, color='red', marker='o', linestyle='-', linewidth=2, label='Predicted 2025 (LR)')
ax2.plot(months_range, predicted_full_year_hw, color='orange', marker='o', linestyle='-', linewidth=2, label='Predicted 2025 (Holt-Winters)')
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
