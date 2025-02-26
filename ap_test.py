import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Load datasets
@st.cache_resource
def load_data():
    global_balance_df = pd.read_csv(r"C:\Users\ialioua\Documents\Crude Demand\GlobalBalance.2025-02-19T09-55.csv")
    refinery_maintenance_df = pd.read_csv(r"C:\Users\ialioua\Documents\Crude Demand\RefineryMaintenance.2025-02-19T09-55.csv")
    return global_balance_df, refinery_maintenance_df

df, df_maintenance = load_data()

# Convert dates
st.title("Crude Demand Forecasting")

st.sidebar.header("Data Preview")
if st.sidebar.checkbox("Show Global Balance Data"):
    st.write(df.head())
if st.sidebar.checkbox("Show Refinery Maintenance Data"):
    st.write(df_maintenance.head())

df["ReferenceDate"] = pd.to_datetime(df["ReferenceDate"], format='%Y-%m-%d')
df_maintenance["StartDate"] = pd.to_datetime(df_maintenance["StartDate"], format='%Y-%m-%d')
df_maintenance["EndDate"] = pd.to_datetime(df_maintenance["EndDate"], format='%Y-%m-%d')

# Create a date range for each maintenance period
maintenance_periods = []
for _, row in df_maintenance.iterrows():
    maintenance_periods.append(pd.DataFrame({
        "ReferenceDate": pd.date_range(row["StartDate"], row["EndDate"], freq='D'),
        "CapacityOffline": row["CapacityOffline"]
    }))

df_maintenance_expanded = pd.concat(maintenance_periods, ignore_index=True)
df_maintenance_expanded["Month"] = df_maintenance_expanded["ReferenceDate"].dt.to_period("M")
monthly_maintenance = df_maintenance_expanded.groupby("Month")["CapacityOffline"].sum().reset_index()
monthly_maintenance["ReferenceDate"] = monthly_maintenance["Month"].dt.to_timestamp()

# Pivot global demand dataset
df_pivot = df.pivot_table(index="ReferenceDate", columns="FlowBreakdown", values="ObservedValue")
df_pivot.columns = [col.replace(" ", "_") for col in df_pivot.columns]

df_pivot = df_pivot.merge(monthly_maintenance[["ReferenceDate", "CapacityOffline"]], on="ReferenceDate", how="left")
df_pivot["CapacityOffline"].fillna(0, inplace=True)

demand_col = "Demand"
if demand_col in df_pivot.columns:
    df_pivot[demand_col] = df_pivot[demand_col].fillna(method="ffill").fillna(method="bfill")

# Define forecasting horizons
horizons = [1, 2, 3, 4, 5, 6]
for horizon in horizons:
    df_pivot[f"Target_{horizon}"] = df_pivot[demand_col].shift(-horizon)

df_pivot.drop(columns=[demand_col], inplace=True)
df_pivot.dropna(inplace=True)

train_size = int(len(df_pivot) * 0.8)
train, test = df_pivot.iloc[:train_size], df_pivot.iloc[train_size:]

X_train = train.drop(columns=[f"Target_{h}" for h in horizons])
X_test = test.drop(columns=[f"Target_{h}" for h in horizons])

y_train_actuals = {h: train[f"Target_{h}"] for h in horizons}
y_test_actuals = {h: test[f"Target_{h}"] for h in horizons}

feature_importance = {}
mae_scores = {}
mape_scores = {}

for horizon in horizons:
    y_train = train[f"Target_{horizon}"]
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    mae_scores[horizon] = mean_absolute_error(y_test_actuals[horizon], y_pred_test)
    mape_scores[horizon] = mean_absolute_percentage_error(y_test_actuals[horizon], y_pred_test)
    feature_importance[horizon] = model.feature_importances_

# Display results
st.header("Model Performance")
st.write("Mean Absolute Error (MAE) per horizon:")
st.write(mae_scores)
st.write("Mean Absolute Percentage Error (MAPE) per horizon:")
st.write(mape_scores)

# Plot feature importance
st.header("Feature Importance")
feature_importance_df = pd.DataFrame(feature_importance, index=X_train.columns)
st.bar_chart(feature_importance_df)


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Load datasets
@st.cache
def load_data():
    global_balance_df = pd.read_csv("/mnt/data/GlobalBalance.2025-02-19T09-55.csv")
    refinery_maintenance_df = pd.read_csv("/mnt/data/RefineryMaintenance.2025-02-19T09-55.csv")
    return global_balance_df, refinery_maintenance_df

df, df_maintenance = load_data()

# Convert dates
st.title("Crude Demand Forecasting")

st.sidebar.header("Data Preview")
if st.sidebar.checkbox("Show Global Balance Data"):
    st.write(df.head())
if st.sidebar.checkbox("Show Refinery Maintenance Data"):
    st.write(df_maintenance.head())

df["ReferenceDate"] = pd.to_datetime(df["ReferenceDate"], format='%Y-%m-%d')
df_maintenance["StartDate"] = pd.to_datetime(df_maintenance["StartDate"], format='%Y-%m-%d')
df_maintenance["EndDate"] = pd.to_datetime(df_maintenance["EndDate"], format='%Y-%m-%d')

# Create a date range for each maintenance period
maintenance_periods = []
for _, row in df_maintenance.iterrows():
    maintenance_periods.append(pd.DataFrame({
        "ReferenceDate": pd.date_range(row["StartDate"], row["EndDate"], freq='D'),
        "CapacityOffline": row["CapacityOffline"]
    }))

df_maintenance_expanded = pd.concat(maintenance_periods, ignore_index=True)
df_maintenance_expanded["Month"] = df_maintenance_expanded["ReferenceDate"].dt.to_period("M")
monthly_maintenance = df_maintenance_expanded.groupby("Month")["CapacityOffline"].sum().reset_index()
monthly_maintenance["ReferenceDate"] = monthly_maintenance["Month"].dt.to_timestamp()

# Pivot global demand dataset
df_pivot = df.pivot_table(index="ReferenceDate", columns="FlowBreakdown", values="ObservedValue")
df_pivot.columns = [col.replace(" ", "_") for col in df_pivot.columns]

df_pivot = df_pivot.merge(monthly_maintenance[["ReferenceDate", "CapacityOffline"]], on="ReferenceDate", how="left")
df_pivot["CapacityOffline"].fillna(0, inplace=True)

demand_col = "Demand"
if demand_col in df_pivot.columns:
    df_pivot[demand_col] = df_pivot[demand_col].fillna(method="ffill").fillna(method="bfill")

# Define forecasting horizons
horizons = [1, 2, 3, 4, 5, 6]
for horizon in horizons:
    df_pivot[f"Target_{horizon}"] = df_pivot[demand_col].shift(-horizon)

df_pivot.drop(columns=[demand_col], inplace=True)
df_pivot.dropna(inplace=True)

train_size = int(len(df_pivot) * 0.8)
train, test = df_pivot.iloc[:train_size], df_pivot.iloc[train_size:]

X_train = train.drop(columns=[f"Target_{h}" for h in horizons])
X_test = test.drop(columns=[f"Target_{h}" for h in horizons])

y_train_actuals = {h: train[f"Target_{h}"] for h in horizons}
y_test_actuals = {h: test[f"Target_{h}"] for h in horizons}

feature_importance = {}
mae_scores = {}
mape_scores = {}

for horizon in horizons:
    y_train = train[f"Target_{horizon}"]
    #model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    #model.fit(X_train, y_train)
    model = joblib.load(f"model_horizon_{horizon}.pkl")

    y_pred_test = model.predict(X_test)
    mae_scores[horizon] = mean_absolute_error(y_test_actuals[horizon], y_pred_test)
    mape_scores[horizon] = mean_absolute_percentage_error(y_test_actuals[horizon], y_pred_test)
    feature_importance[horizon] = model.feature_importances_

# Display results
st.header("Model Performance")
st.write("Mean Absolute Error (MAE) per horizon:")
st.write(mae_scores)
st.write("Mean Absolute Percentage Error (MAPE) per horizon:")
st.write(mape_scores)

# Plot feature importance
st.header("Feature Importance")
feature_importance_df = pd.DataFrame(feature_importance, index=X_train.columns)
st.bar_chart(feature_importance_df)
