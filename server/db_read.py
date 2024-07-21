import sqlite3

def get_values_from_db(model_name):
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM predictions WHERE Model=?", (model_name,))

    rows = cursor.fetchall()

    if rows:
        # Extract values from the rows
        filename_values = [str(row[0]) for row in rows]  
        date_values = [str(row[2]) for row in rows] 
        prediction_values = [float(row[3]) for row in rows] 
        test_values = [float(row[4]) for row in rows]  
        mae_values = [float(row[5]) for row in rows]  
        mse_values = [float(row[6]) for row in rows]  
        rmse_values = [float(row[7]) for row in rows]  
        r2_values = [float(row[8]) for row in rows]    

        # Return the values
        return filename_values[0], date_values, prediction_values, test_values, mae_values[0], mse_values[0], rmse_values[0], r2_values[0]
    else:
        print(f"No rows found for model '{model_name}'")
        return None
