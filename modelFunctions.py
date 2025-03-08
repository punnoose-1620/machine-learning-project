import numpy as np
import pandas as pd
import xgboost as xgb
from dataVisualisation import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def convertDateColumn(df: pd.DataFrame, dateKey: str):
    df[dateKey] = pd.to_datetime(df[dateKey])  # Ensure correct datetime format
    df['hour'] = df[dateKey].dt.hour
    df['month'] = df[dateKey].dt.month
    df['year'] = df[dateKey].dt.year

    # Drop original 'date' column (but not sorting the data)
    df = df.drop(columns=[dateKey])
    return df

def getRandomForestModel(nEstimators: int = 100, randomState: int = 42):
    """
    Function creates and returns **Random Forest Regression** model of type `sklearn.ensemble.RandomForestRegressor`.\n
    """
    rf = RandomForestRegressor(n_estimators=nEstimators, random_state=randomState)
    return rf

def getPolynomialRegressionModel():
    """
    Function creates and returns **Polynomial Regression** model of type `sklearn.linear_model.LinearRegression`.\n
    """
    pr = LinearRegression()
    return pr

def getXgBoostRegressionModel(nEstimators: int = 100, learningRate: float = 0.1, randomState:int = 42):
    xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
    return xgb

def getAnnRegressionModel(X_train_scaled: np.ndarray):
    ANN = Sequential([
        Input(shape=(X_train_scaled.shape[1],)),  # Input layer (feature size must match X_train)
        Dense(64, activation='relu'),  # First hidden layer with ReLU activation
        Dense(32, activation='relu'),  # Second hidden layer
        Dense(1)  # Output layer (for regression, no activation function)
    ])
    return ANN

def rfTrainer(
        model, 
        filePath: str, 
        dateKey: str, 
        targetKey: str, 
        testSize: float = 0.2, 
        randomState: int = 42
        ):
    """
    Function for training and testing Random Forest Regression.\n

    Reads data from given file path.\n
    
    Trains and Tests the model on the given data.\n
    
    Plots all related graphs and Saves Graphs to GraphPlots folder.\n
    
    Parameters :
    - **model** -> Regression Model, of type `sklearn.ensemble.RandomForestRegressor`.
    - **filePath** -> Path to CSV data file, of type `str`.
    - **dateKey** -> Key used to specify date and time, of type `str`.
    - **targetKey** -> Key used to specify target column, of type `str`.
    - **testSize** -> Percentage of data to be used for testing, of type `float` in range 0.0 to 1.0.
    - **randomState** -> RandomState value to be used for Model, of type `int`.
    """
    df = pd.read_csv(filePath)        # Load the data from file
    
    # Convert 'date' column to datetime and extract features
    df = convertDateColumn(df, dateKey)

    # Split data into features (X) and target (y)
    X = df.drop(columns=[targetKey])  
    y = df[targetKey]

    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState)
    
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Compute evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print evaluation results
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R² Score: {r2:.4f}")

    # Plot Results of the Model
    plotRfResults(
        model=model, 
        y_test=y_test, 
        y_pred=y_pred, 
        )
    
def prTrainer(
        model: LinearRegression,
        filePath: str, 
        dateKey: str, 
        targetKey: str, 
        testSize: float = 0.2, 
        randomState: int = 42
        ):
    '''Function for training and testing Polynomial Regression.\n
    Reads data from given file path.\n
    Takes given model. Trains data on the model.\n
    Tests data on the model.\n
    Plots all related graphs.\n
    Saves Graphs to GraphPlots folder.\n
    Parameters :
    - **model** -> Regression Model, of type `sklearn.linear_model.LinearRegression`.
    - **filePath** -> Path to CSV data file, of type `str`.
    - **dateKey** -> Key used to specify date and time, of type `str`.
    - **targetKey** -> Key used to specify target column, of type `str`.
    - **testSize** -> Percentage of data to be used for testing, of type `float` in range 0.0 to 1.0.
    - **randomState** -> RandomState value to be used for Model, of type `int`.
    '''
    # Load the dataset
    df = pd.read_csv(filePath)

    # Convert 'date' column to datetime and extract features
    df = convertDateColumn(df, dateKey)

    # Split data into features (X) and target (y)
    X = df.drop(columns=[targetKey])  
    y = df[targetKey]

    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState)

    # Apply Polynomial Features Transformation (Degree 2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Scale the features (important for polynomial regression)
    scaler = StandardScaler()
    X_train_poly = scaler.fit_transform(X_train_poly)
    X_test_poly = scaler.transform(X_test_poly)

    # Train Polynomial Regression (Linear Regression on Polynomial Features)
    model = getPolynomialRegressionModel()
    model.fit(X_train_poly, y_train)

    # Predict on test data
    y_pred = model.predict(X_test_poly)

    # Compute evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print evaluation results
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R² Score: {r2:.4f}")

    # Plot Results of the Model
    plotRfResults(
        y_test=y_test, 
        y_pred=y_pred, 
        )

def xgbTrainer(
        model: xgb.XGBRegressor,
        filePath: str, 
        dateKey: str, 
        targetKey: str, 
        testSize: float = 0.2, 
        randomState: int = 42
        ):
    '''Function for training and testing XGBoost Regression.\n
    Reads data from given file path.\n
    Takes given model. Trains data on the model.\n
    Tests data on the model.\n
    Plots all related graphs.\n
    Saves Graphs to GraphPlots folder.\n
    Parameters :
    - **model** -> XGBoost Model, of type `xgboost.XGBRegressor`.
    - **filePath** -> Path to CSV data file, of type `str`.
    - **dateKey** -> Key used to specify date and time, of type `str`.
    - **targetKey** -> Key used to specify target column, of type `str`.
    - **testSize** -> Percentage of data to be used for testing, of type `float` in range 0.0 to 1.0.
    - **randomState** -> RandomState value to be used for Model, of type `int`.
    '''
    # Load the dataset
    df = pd.read_csv(filePath)

    # Convert 'date' column to datetime and extract features
    df = convertDateColumn(df, dateKey)

    # Split data into features (X) and target (y)
    X = df.drop(columns=[targetKey])  
    y = df[targetKey]

    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState)

    # Train XGBoost Regressor
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Compute evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print evaluation results
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R² Score: {r2:.4f}")

    # Plot Results of the Model
    plotXgbResults(
        model=model,
        X=X,
        y_test=y_test, 
        y_pred=y_pred
        )
    
def AnnTrainer(
        model: Sequential, 
        filePath: str, 
        dateKey: str, 
        targetKey: str, 
        testSize: float = 0.2, 
        randomState: int = 42,
        epochs: int = 50,
        batchSize: int = 32,
        optimizer: str = 'adam',
        lossFunction: str = 'mse'
        ):
    '''Function for training and testing XGBoost Regression.\n
    Reads data from given file path.\n
    Takes given model. Trains data on the model.\n
    Tests data on the model.\n
    Plots all related graphs.\n
    Saves Graphs to GraphPlots folder.\n
    Parameters :
    - **model** -> Artificial Neural Network Model, of type `tensorflow.keras.models.Sequential`.
    - **filePath** -> Path to CSV data file, of type `str`.
    - **dateKey** -> Key used to specify date and time, of type `str`.
    - **targetKey** -> Key used to specify target column, of type `str`.
    - **testSize** -> Percentage of data to be used for testing, of type `float` in range 0.0 to 1.0.
    - **randomState** -> RandomState value to be used for Model, of type `int`.
    - **epochs** -> Number of total epochs to be run, of type `int`
    - **batchSize** -> Batch size to be used for cross validation, of type `int`
    - **optimizer** -> Optimizer function to be used, of type `str`
    - **lossFunction** -> Loss function to be used to evaluate model, of type `str`
    '''
    # Load the dataset
    df = pd.read_csv(filePath)

    # Convert 'date' column to datetime and extract features
    df = convertDateColumn(df, dateKey)

    # Split data into features (X) and target (y)
    X = df.drop(columns=[targetKey])  
    y = df[targetKey]

    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState)

    # Standardize the features (important for ANN training)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Compile the model
    model.compile(optimizer=optimizer, loss=lossFunction, metrics=['mae'])

    # Train the model
    history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batchSize, validation_data=(X_test_scaled, y_test))

    # Predict on test data
    y_pred = model.predict(X_test_scaled).flatten()

    # Compute evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print evaluation results
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R² Score: {r2:.4f}")

    # Plot Results of the Model
    plotAnnResults(
        y_test=y_test,
        y_pred=y_pred,
        history=history.history
    )