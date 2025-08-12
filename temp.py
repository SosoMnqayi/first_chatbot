import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


st.title('ARIMA Time Series Forecasting App')
st.write('Upload a CSV file with your time series data and select the target column for forecasting.')

# Upload a CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)


    st.write('Uploaded Data:')
    st.write(df)

   
    target_column = st.selectbox('Select the target column for forecasting', df.columns)

   
    p, d, q = st.slider('Select ARIMA Order (p, d, q)', 0, 10, (1, 1, 1))
    model = ARIMA(df[target_column], order=(p, d, q))
    model_fit = model.fit()

    
    forecast_steps = st.slider('Select the number of forecasted time steps', 1, 100, 10)
    forecast = model_fit.forecast(steps=forecast_steps)

    # Create a time index for the forecasted values
    forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, closed='right')[1:]

   
    forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_index)

    # Plot the original data and the forecasted values
    st.write('ARIMA Forecast:')
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[target_column], label='Original Data')
    plt.plot(forecast_df.index, forecast_df['Forecast'], label='ARIMA Forecast', linestyle='--', color='green')
    plt.xlabel('Date')
    plt.ylabel(target_column)
    plt.title(f'ARIMA Forecast for {target_column}')
    plt.legend()
    st.pyplot()

    # Calculate the Mean Squared Error (MSE) for the forecast
    mse = mean_squared_error(df[target_column][-forecast_steps:], forecast)
    st.write(f'Mean Squared Error (MSE): {mse}')
