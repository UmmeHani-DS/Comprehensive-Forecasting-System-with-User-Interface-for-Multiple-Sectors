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
from tensorflow.keras.models import Sequential, load_model# type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tqdm import tqdm
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# Load the ARIMA model from the pickle file
with open('arima_model.pkl', 'rb') as f:
    arima_model = pickle.load(f)

with open('sarima_model.pkl', 'rb') as f:
    sarima_model = pickle.load(f)

with open('ses_model.pkl', 'rb') as f:
    ses_model = pickle.load(f) 

with open('prophet_model.pkl', 'rb') as f:
    prophet_model = pickle.load(f)

with open('svr_model.pkl', 'rb') as f:
    svr_model = pickle.load(f)

lstm_model = load_model('lstm_model.keras')

ANN_model = load_model('ANN_model.keras')

hybrid_predictions = pd.read_csv('Hybrid_results.csv')

filename = 'co2_trend_gl.csv'

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def insert_into_db(filename, model_name, date_values, prediction_values, test_values, mae, mse, rmse, r2):
    conn = sqlite3.connect('predictions.db')
    print('Connected To DB')
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    conn.execute('CREATE TABLE IF NOT EXISTS predictions (filename TEXT, Model TEXT, Date INTEGER, prediction REAL, test REAL, MAE REAL, MSE REAL, RMSE REAL, R2 REAL)')
    conn.commit()

    print('Model: ', model_name)
    for prediction, test, date in zip(prediction_values, test_values, date_values):
        cursor.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?)", (filename, model_name, date, prediction, test, mae, mse, rmse, r2))

    print('Inserted Successfully!')

    conn.commit()
    conn.close()

def ARIMA_insert():

    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df.set_index('date')

    train = df['smoothed'][:-200]
    test = df['smoothed'][-200:] 

    formatted_actual_dates = test.index.strftime('%Y-%m-%d').tolist()

    predictions = arima_model.predict(start=len(train), end=len(train) + len(test) - 1)
    predictions.index = test.index 

    mae = mean_absolute_error(test, predictions)
    mse = mean_squared_error(test, predictions)
    rmse = sqrt(mse)
    r2 = r2_score(test, predictions)

    insert_into_db(filename, 'ARIMA', formatted_actual_dates, predictions, test, mae, mse, rmse, r2)

def SARIMA_insert():

    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df.set_index('date')

    train = df['smoothed'][:-100]
    test = df['smoothed'][-100:]

    predictions = pd.Series(sarima_model.predict(n_periods=len(test)))
    predictions.index = test.index  

    formatted_actual_dates = test.index.strftime('%Y-%m-%d').tolist()

    mae = mean_absolute_error(test, predictions)
    mse = mean_squared_error(test, predictions)
    rmse = sqrt(mse)
    r2 = r2_score(test, predictions) 

    insert_into_db(filename, 'SARIMA', formatted_actual_dates, predictions, test, mae, mse, rmse, r2) 

def SES_insert():

    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df.set_index('date')

    formatted_actual_dates = df.index.strftime('%Y-%m-%d').tolist()

    ses_predictions = ses_model.predict(start=df.index[0], end=df.index[-1])
    ses_mse = mean_squared_error(df['smoothed'], ses_predictions)
    ses_mae = mean_absolute_error(df['smoothed'], ses_predictions)
    ses_mape = mean_absolute_percentage_error(df['smoothed'], ses_predictions)
    ses_r2 = r2_score(df['smoothed'], ses_predictions)

    insert_into_db(filename, 'SES', formatted_actual_dates, ses_predictions, df['smoothed'], ses_mae, ses_mse, sqrt(ses_mse), ses_r2)

def prophet_insert():

    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df.set_index('date')

    train = df['smoothed'][:-500]
    test = df['smoothed'][-500:]

    formatted_actual_dates = test.index.strftime('%Y-%m-%d').tolist()

    future = prophet_model.make_future_dataframe(periods=len(test))
    forecast = prophet_model.predict(future)

    forecast_subset = forecast.set_index('ds').loc[test.index]
    prophet_mse = mean_squared_error(test, forecast_subset['yhat'])
    prophet_mae = mean_absolute_error(test, forecast_subset['yhat'])
    prophet_mape = mean_absolute_percentage_error(test, forecast_subset['yhat'])
    prophet_r2 = r2_score(test, forecast_subset['yhat'])

    insert_into_db(filename, 'Prophet', formatted_actual_dates, forecast_subset['yhat'], test, prophet_mae, prophet_mse, sqrt(prophet_mse), prophet_r2)

def SVR_insert():

    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df.set_index('date')

    X = df.drop(columns=['smoothed'])  # Features
    y = df['smoothed']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_pred = svr_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    formatted_actual_dates = y_test.index.strftime('%Y-%m-%d').tolist()

    insert_into_db(filename, 'SVR', formatted_actual_dates, y_pred, y_test, mean_absolute_error(y_test, y_pred), mse, rmse, r2)

def LSTM_insert():

    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df.set_index('date')

    train_size = int(len(df) * 0.8)  # 80% for training, 20% for testing
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    scaler = MinMaxScaler()

    scaled_train = scaler.fit_transform(train[['smoothed']])
    scaled_test = scaler.transform(test[['smoothed']])

    n_input = 20

    n_features = 1
    test_generator = TimeseriesGenerator(scaled_test, scaled_test, length=n_input, batch_size=1)

    test_preds = lstm_model.predict(test_generator)

    test_preds_inv = scaler.inverse_transform(test_preds).flatten()
    actual_values = test['smoothed'].values[n_input:]

    lstm_mse = mean_squared_error(actual_values, test_preds_inv)
    lstm_mae = mean_absolute_error(actual_values, test_preds_inv)
    lstm_mape = mean_absolute_percentage_error(actual_values, test_preds_inv)
    lstm_r2 = r2_score(actual_values, test_preds_inv)

    formatted_actual_dates = test.index.strftime('%Y-%m-%d').tolist()

    insert_into_db(filename, 'LSTM', formatted_actual_dates, test_preds_inv.tolist(), actual_values, lstm_mse, lstm_mae, lstm_mape, lstm_r2)

def ANN_insert():

    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df.set_index('date')

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['smoothed']])

    n_steps = 50

    X = []
    y = []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i - n_steps:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    predictions = ANN_model.predict(X_val)

    predictions_inv = scaler.inverse_transform(predictions)

    ann_mse = mean_squared_error(y_val, predictions_inv)
    ann_mae = mean_absolute_error(y_val, predictions_inv)
    ann_mape = mean_absolute_percentage_error(y_val, predictions_inv)
    ann_r2 = r2_score(y_val, predictions_inv)

    dates_val = df.index[split + n_steps:]

    y_val = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

    formatted_actual_dates = dates_val.strftime('%Y-%m-%d').tolist()

    insert_into_db(filename, 'ANN', formatted_actual_dates, predictions_inv.flatten().tolist(), y_val.tolist(), ann_mse, ann_mae, ann_mape, ann_r2)   

def Hybrid_insert():

    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df.set_index('date')

    train = df['smoothed'][:-200]
    test = df['smoothed'][-200:] 

    arima_predictions = arima_model.predict(start=len(train), end=len(train) + len(test) - 1)
    arima_predictions = arima_predictions.values.reshape(-1, 1)

    residuals = test.values - arima_predictions.flatten()

    n_features = ANN_model.input_shape[1]
    repeated_predictions = np.repeat(arima_predictions, n_features, axis=1)

    ANN_model.fit(repeated_predictions, residuals, epochs=100, batch_size=32)

    hybrid_residuals = ANN_model.predict(repeated_predictions)

    hybrid_predictions = arima_predictions.flatten() + hybrid_residuals.flatten()

    hybrid_mae = mean_absolute_error(test, hybrid_predictions)
    hybrid_mse = mean_squared_error(test, hybrid_predictions)
    hybrid_rmse = np.sqrt(hybrid_mse)
    hybrid_r2 = r2_score(test, hybrid_predictions)

    formatted_actual_dates = test.index.strftime('%Y-%m-%d').tolist()

    insert_into_db(filename, 'Hybrid', formatted_actual_dates, hybrid_predictions, test, hybrid_mae, hybrid_mse, hybrid_rmse, hybrid_r2)

if __name__ == '__main__':
    ARIMA_insert()
    SARIMA_insert()
    SES_insert()
    prophet_insert()
    SVR_insert()
    LSTM_insert()
    ANN_insert()
    Hybrid_insert()
