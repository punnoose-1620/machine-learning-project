import joblib
import pandas as pd
from modelFunctions import *
from dataVisualisation import *
from tensorflow.keras.models import load_model      # type: ignore

modelPath = './ResultModels/XGBoost_Model.pkl'
date_key = 'timestamp'
target_key = 'price'
export_key = 'export'

def readCsv(filePath: str, targetColumn: str):
    data = pd.read_csv(filePath)
    
    # Drop rows with missing target values
    data = data.dropna(subset=[targetColumn])
    return data

def load_model_and_predict(filename, X_test, targetKey: str = 'price'):
    """
    Load model from file and predict on new data.

    Args:
        filename (str): Path to saved model.
        X_test (array-like): Test data for prediction.

    Returns:
        y_pred: Predicted values.
    """
    if('.pkl' in filename):
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")

        if targetKey in X_test.columns:
            X_test.drop(columns=[targetKey], inplace=True)
        
        print(f'\nX_Test : {x_test}\n')
        
        y_pred = model.predict(X_test)
        return y_pred
    else:
        best_model = load_model(filename)
        y_pred = best_model.predict(X_test)

def get_last_7_days(filename: str, targetKey: str = 'price', dateKey: str = 'timestamp'):

    entries_in_7_days = 7*24
    
    df = readCsv(filename, targetKey)

    # X value with original timestamps
    X_test_originals = df.drop(columns=[targetKey])  

    df = convert_column_to_float_and_drop_invalid(df=df, column_name=targetKey)

    # Convert 'date' column to datetime and extract features
    df = convertDateColumn(df, dateKey, '')

    # Split data into features (X) and target (y)
    X_test = df.drop(columns=[targetKey])  
    y_test = df[targetKey]

    previous_week_data = X_test.iloc[(-2*entries_in_7_days):-entries_in_7_days]
    previous_week_export_avg = previous_week_data[export_key].mean()

    # Replace last 7 values in 'export' with value at 9th from last
    X_test.loc[X_test.index[-entries_in_7_days:], export_key] = previous_week_export_avg

    current_week_data = df.tail(entries_in_7_days)
    y_test = y_test.tail(entries_in_7_days)

    return current_week_data, y_test, X_test_originals

x_test, y_test, original_test = get_last_7_days(finalMergedFile, targetKey=target_key, dateKey=date_key)
y_pred = load_model_and_predict(filename=modelPath, X_test=x_test, targetKey=target_key)
# print(f"\nLength Check :\ny_test({len(y_test)}) : {y_test}\ny_pred({len(y_pred)}) : {y_pred}\n")
plot_predictions_over_time(y_pred=y_pred, y_test=y_test, X_test=original_test, datetime_column_name=date_key)

differences = [round(abs(y_t - y_p), ndigits=2) for y_t, y_p in zip(y_test, y_pred)]
lowest = 250.0
highest = 0.0
mean = 0.0
for value in differences:
    if value<lowest:
        lowest = value
    if value>highest:
        highest = value
    mean = round((mean+value)/2, ndigits=2)

print(f'\nLowest Difference : {lowest} SEK')
print(f'Highest Difference : {highest} SEK')
print(f'Average Difference : {mean} SEK\n')

# Compute evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Evaluation Metrics for last week predictions : \nMSE : {mse}\nMAE : {mae}\nR2 : {r2}")