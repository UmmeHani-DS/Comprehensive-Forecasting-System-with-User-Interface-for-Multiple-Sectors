import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import itertools
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import statsmodels.api as sm
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.svm import SVR
import pickle
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator # type: ignore
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tqdm import tqdm
import sqlite3
import warnings
warnings.filterwarnings('ignore')

filename = 'co2_trend_gl.csv'
df = pd.read_csv(filename)

df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
df = df.set_index('date')

def get_forecast_Arima(no_of_days, arima_model):

    predictions = arima_model.forecast(steps=no_of_days)
    predictions = pd.Series(predictions)
    predictions.index = pd.date_range(start=df.index[-1], periods=no_of_days, freq=df.index.freq)

    return predictions

def get_forecast_Sarima(no_of_days, sarima_model):

    test = df['smoothed'][-100:]

    forecast = pd.Series(sarima_model.predict(n_periods=no_of_days))

    last_index = test.index[-1]
    forecast_index = pd.date_range(start=last_index, periods=no_of_days)

    forecast.index = forecast_index

    return forecast

def get_forecast_SES(no_of_days, ses_result):

    ses_forecast = ses_result.forecast(steps=no_of_days)
    ses_forecast = pd.Series(ses_forecast)
    ses_forecast.index = pd.date_range(start=df.index[-1], periods=no_of_days, freq=df.index.freq)

    return ses_forecast

def get_forecast_prophet(no_of_days, prophet_model):

    test = df['smoothed'][-500:]

    total_future_periods = len(test) + no_of_days
    future = prophet_model.make_future_dataframe(periods=total_future_periods)

    forecast = prophet_model.predict(future)

    future_forecast = forecast.set_index('ds').loc[test.index[-1]:]
    future_forecast_200 = future_forecast.iloc[1:201]

    return future_forecast_200['yhat']

def get_forecast_LSTM(no_of_days, lstm_model):

    future_steps = no_of_days

    train_size = int(len(df) * 0.8)  # 80% for training, 20% for testing
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    scaler = MinMaxScaler()

    scaled_train = scaler.fit_transform(train[['smoothed']])
    scaled_test = scaler.transform(test[['smoothed']])

    n_input = 20
    n_features = 1

    initial_input = scaled_test[-n_input:].reshape((1, n_input, n_features))

    future_preds = []

    for i in range(future_steps):
        future_pred = lstm_model.predict(initial_input)[0, 0]
        future_preds.append(future_pred)
        
        # Update the input sequence by removing the first value and appending the predicted value
        future_input = np.append(initial_input[:, 1:, :], [[[future_pred]]], axis=1)
        initial_input = future_input

    future_preds_inv = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date, periods=future_steps + 1)[1:]

    return future_dates, future_preds_inv

def forecast_future(model, initial_input, scaler, n_forecast_days):
    forecast = []

    # Initial input sequence
    current_input = initial_input.reshape(1, -1)  # Reshape for model prediction

    # Forecast for n_forecast_days
    for _ in range(n_forecast_days):
        # Predict the next day's value
        next_day_prediction = model.predict(current_input)
        
        # Inverse transform the prediction
        next_day_prediction_inv = scaler.inverse_transform(next_day_prediction)[0][0]
        
        # Append the prediction to the forecast
        forecast.append(next_day_prediction_inv)

        # Update the input sequence for the next prediction
        current_input = np.append(current_input[:, 1:], next_day_prediction).reshape(1, -1)

    return forecast

def get_forecast_ANN(no_of_days, ANN_model):

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['smoothed']])

    n_steps = 50

    forecast_dates = pd.date_range(start=df.index[-1], periods=no_of_days)
    initial_input = scaled_data[-n_steps:, 0]
    forecast = forecast_future(ANN_model, initial_input, scaler, no_of_days)

    forecast_df = pd.DataFrame(forecast, index=forecast_dates, columns=['Forecasted'])

    return forecast_df
