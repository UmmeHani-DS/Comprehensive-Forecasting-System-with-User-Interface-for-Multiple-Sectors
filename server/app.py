from flask import Flask, request, jsonify
from flask_cors import CORS # type: ignore
import pandas as pd
from db_read import get_values_from_db
from db_insert import *
from get_forecast import *

app = Flask(__name__)
CORS(app)

filename = 'co2_trend_gl.csv'

@app.route('/data', methods=['GET'])
def get_data():
    # Reset future_dates at the beginning of each request
    global future_dates
    future_dates = None

    # Read the CSV file and preprocess the data
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df.set_index('date')
    formatted_actual_dates = df.index.strftime('%Y-%m-%d').tolist()

    df_dict = df.to_dict(orient='records')

    forecast = None

    # Get the selected model and date from the query parameters
    selected_model = request.args.get('model', None)
    selected_date = request.args.get('date', None)
    selected_date = int(selected_date)

    if selected_model == 'ARIMA':
        forecast = get_forecast_Arima(selected_date, arima_model)
    elif selected_model == 'SARIMA':
        forecast = get_forecast_Sarima(selected_date, sarima_model)
    elif selected_model == 'SES':
        forecast = get_forecast_SES(selected_date, ses_model)
    elif selected_model == 'Prophet':
        forecast = get_forecast_prophet(selected_date, prophet_model)
    elif selected_model == 'LSTM':
        future_dates, forecast = get_forecast_LSTM(selected_date, lstm_model)
    elif selected_model == 'ANN':
        forecast = get_forecast_ANN(selected_date, ANN_model)

    # Get values from the database for the selected model and date range
    filename_values, date_values, prediction_values, test_values, mae_values, mse_values, rmse_values, r2_values = get_values_from_db(selected_model)

    if selected_model == 'LSTM' and future_dates is not None:
        future_dates = future_dates.strftime('%Y-%m-%d').tolist()
        forecast = forecast.flatten().tolist()

        # Return the fetched values along with preprocessed data through Flask
        return jsonify({
            'filename_values': filename_values,
            'date_values': date_values,
            'prediction_values': prediction_values,
            'test_values': test_values,
            'mae_values': mae_values,
            'mse_values': mse_values,
            'rmse_values': rmse_values,
            'r2_values': r2_values,
            'formatted_actual_dates': formatted_actual_dates,
            'actual_values': df_dict,
            'future_dates': future_dates,
            'forecast': forecast
        })
    else:
        # Handle case where future_dates is None or empty or when model is not LSTM
        if hasattr(forecast, 'index') and hasattr(forecast, 'values'):
            forecast_index = forecast.index.strftime('%Y-%m-%d').tolist()
            forecast_values = forecast.values.flatten().tolist()
        else:
            forecast_index = []
            forecast_values = []

        # Return the fetched values along with preprocessed data through Flask
        return jsonify({
            'filename_values': filename_values,
            'date_values': date_values,
            'prediction_values': prediction_values,
            'test_values': test_values,
            'mae_values': mae_values,
            'mse_values': mse_values,
            'rmse_values': rmse_values,
            'r2_values': r2_values,
            'formatted_actual_dates': formatted_actual_dates,
            'actual_values': df_dict,
            'future_dates': forecast_index,
            'forecast': forecast_values
        })

if __name__ == '__main__':
    app.run(debug=True, port=8080)
