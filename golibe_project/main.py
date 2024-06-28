import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import STLForecast
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf

st.title("Studying Energy Consumption with Time Series Analysis")


# Load data
def load_data():
    file = "./data/consumption_cleaned.csv"
    data = pd.read_csv(file, index_col=0)
    data.columns = pd.to_datetime(data.columns, format="%b%y")
    return data


data = load_data()

st.subheader("Recorded Residents")
st.write(data.head())

st.subheader("Visualizing Data")
fig = px.line(
    data.sample(10).T,
    title=f"Energy Consumption for 10 sampled residents",
)
st.plotly_chart(fig)

fig = px.line(
    data.mean(),
    title=f"Average Consumption for all residents across time",
)
st.plotly_chart(fig)

# Autocorrelation Plot
st.subheader("Autocorrelation Plot (ACF)")
fig, ax = plt.subplots()
ax.set_facecolor("#0e1117")
autocorrelation = data.mean(axis=0).autocorr()
fig.patch.set_alpha(0.0)
plot_acf(data.mean(axis=0), ax=ax)
ax.tick_params(colors="white")
ax.yaxis.label.set_color("white")
ax.xaxis.label.set_color("white")
for spine in ax.spines.values():
    spine.set_edgecolor("white")
st.pyplot(fig)

st.subheader("Individual Resident Analysis")
person = st.selectbox("Select resident for analysis", data.index)

# Plot historical data
st.subheader("Historical Data")
fig = px.line(data.T, y=person, title=f"Energy Consumption of {person}")
st.plotly_chart(fig)

# Additional data visualizations
st.subheader("Additional Data Visualizations")

# Monthly average consumption
monthly_avg = data.mean(axis=0)
fig = px.line(monthly_avg, title="Monthly Average Energy Consumption")
st.plotly_chart(fig)

# Yearly average consumption
yearly_avg = data.T.resample("Y").mean().T
fig = px.box(yearly_avg, title="Yearly Average Energy Consumption")
st.plotly_chart(fig)

# Distribution of consumption
fig = px.histogram(data.T, title="Distribution of Energy Consumption")
st.plotly_chart(fig)

# Prepare data for time series analysis
ts_data = data.T[person]

# Split data into training and testing sets
train_size = int(len(ts_data) * 0.8)
train, test = ts_data[:train_size], ts_data[train_size:]

# ARIMA Model
st.subheader("ARIMA Model")
arima_model = ARIMA(train, order=(5, 1, 0))
arima_result = arima_model.fit()
arima_forecast = arima_result.forecast(steps=len(test))

# Plot ARIMA results
fig, ax = plt.subplots()
ax.set_facecolor("#0e1117")
fig.patch.set_alpha(0.0)
ax.tick_params(colors="white")
ax.yaxis.label.set_color("white")
ax.xaxis.label.set_color("white")
for spine in ax.spines.values():
    spine.set_edgecolor("white")

ax.plot(train.index, train, label="Train")
ax.plot(test.index, test, label="Test")
ax.plot(test.index, arima_forecast, label="ARIMA Forecast")
ax.legend()
st.pyplot(fig)

arima_mae = mean_absolute_error(test, arima_forecast)
st.write(f"ARIMA MAE: {arima_mae}")

# Exponential Smoothing (ETS) Model
st.subheader("ETS Model")
ets_model = ExponentialSmoothing(train, seasonal="add", seasonal_periods=5)
ets_result = ets_model.fit()
ets_forecast = ets_result.forecast(steps=len(test))

# Plot ETS results
fig, ax = plt.subplots()
ax.set_facecolor("#0e1117")
fig.patch.set_alpha(0.0)
ax.tick_params(colors="white")
ax.yaxis.label.set_color("white")
ax.xaxis.label.set_color("white")
for spine in ax.spines.values():
    spine.set_edgecolor("white")
ax.plot(train.index, train, label="Train")
ax.plot(test.index, test, label="Test")
ax.plot(test.index, ets_forecast, label="ETS Forecast")
ax.legend()
st.pyplot(fig)

ets_mae = mean_absolute_error(test, ets_forecast)
st.write(f"ETS MAE: {ets_mae}")

# STL + ARIMA Model
st.subheader("STL + ARIMA Model")
stlf_arima = STLForecast(train, ARIMA, model_kwargs={"order": (2, 1, 0)})
stlf_arima_result = stlf_arima.fit()
stlf_arima_forecast = stlf_arima_result.forecast(steps=len(test))

# Plot STL + ARIMA results
fig, ax = plt.subplots()

ax.set_facecolor("#0e1117")
fig.patch.set_alpha(0.0)
ax.tick_params(colors="white")
ax.yaxis.label.set_color("white")
ax.xaxis.label.set_color("white")
for spine in ax.spines.values():
    spine.set_edgecolor("white")

ax.plot(train.index, train, label="Train")
ax.plot(test.index, test, label="Test")
ax.plot(test.index, stlf_arima_forecast, label="STL + ARIMA Forecast")
ax.legend()
st.pyplot(fig)

stlf_arima_mae = mean_absolute_error(test, stlf_arima_forecast)
st.write(f"STL + ARIMA MAE: {stlf_arima_mae}")

# STL + ETS Model
st.subheader("STL + ETS Model")
stlf_ets = STLForecast(
    train,
    ExponentialSmoothing,
    model_kwargs={"seasonal": "add", "seasonal_periods": 12},
)
stlf_ets_result = stlf_ets.fit()
stlf_ets_forecast = stlf_ets_result.forecast(steps=len(test))

# Plot STL + ETS results
fig, ax = plt.subplots()

ax.set_facecolor("#0e1117")
fig.patch.set_alpha(0.0)
ax.tick_params(colors="white")
ax.yaxis.label.set_color("white")
ax.xaxis.label.set_color("white")
for spine in ax.spines.values():
    spine.set_edgecolor("white")
ax.plot(train.index, train, label="Train")
ax.plot(test.index, test, label="Test")
ax.plot(test.index, stlf_ets_forecast, label="STL + ETS Forecast")
ax.legend()
st.pyplot(fig)

stlf_ets_mae = mean_absolute_error(test, stlf_ets_forecast)
st.write(f"STL + ETS MAE: {stlf_ets_mae}")

# Error Analysis
st.subheader("Error Analysis")

# Calculate errors
arima_error = test - arima_forecast
ets_error = test - ets_forecast
stlf_arima_error = test - stlf_arima_forecast
stlf_ets_error = test - stlf_ets_forecast

# Create a DataFrame for errors
error_df = pd.DataFrame(
    {
        "Date": test.index,
        "ARIMA Error": arima_error,
        "ETS Error": ets_error,
        "STL + ARIMA Error": stlf_arima_error,
        "STL + ETS Error": stlf_ets_error,
    }
)

# Plot errors
fig = px.line(
    error_df,
    x="Date",
    y=["ARIMA Error", "ETS Error", "STL + ARIMA Error", "STL + ETS Error"],
    title="Model Prediction Errors",
)
st.plotly_chart(fig)

# Display error statistics
st.write("Error Statistics")
st.write(error_df.describe())

# Future Predictions
st.subheader("Future Predictions")

prediction_range = st.slider("Select the number of months to predict", 1, 24, 12)

# Predict future using ARIMA
future_arima = arima_result.forecast(steps=prediction_range)
# Predict future using ETS
future_ets = ets_result.forecast(steps=prediction_range)
# Predict future using STL + ARIMA
future_stlf_arima = stlf_arima_result.forecast(steps=prediction_range)
# Predict future using STL + ETS
future_stlf_ets = stlf_ets_result.forecast(steps=prediction_range)

# Combine future predictions into a DataFrame
future_dates = pd.date_range(
    start=ts_data.index[-1], periods=prediction_range + 1, freq="M"
)[1:]
future_df = pd.DataFrame(
    {
        "Date": future_dates,
        "ARIMA Forecast": future_arima,
        "ETS Forecast": future_ets,
        "STL + ARIMA Forecast": future_stlf_arima,
        "STL + ETS Forecast": future_stlf_ets,
    }
)
st.write(future_df)

# Plot future predictions
fig = px.line(
    future_df,
    x="Date",
    y=["ARIMA Forecast", "ETS Forecast", "STL + ARIMA Forecast", "STL + ETS Forecast"],
    title="Future Predictions",
)
st.plotly_chart(fig)
