import json
import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from dataVisualisation import *
from scipy.stats import randint
from tensorflow.keras.models import Sequential                      # type: ignore
from tensorflow.keras.layers import Dense, Input                    # type: ignore
from scikeras.wrappers import KerasRegressor                        # type: ignore
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer

print("GPU Available:", tf.config.list_physical_devices('GPU'))

def scaleData(input):
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    input_scaled = scaler.fit_transform(input)
    return input_scaled

def convert_column_to_float_and_drop_invalid(df, column_name):
    """
    Converts entries in a specified column to float.
    Drops rows where conversion fails.

    Args:
        df (pd.DataFrame): The DataFrame.
        column_name (str): Name of the column to convert.

    Returns:
        pd.DataFrame: Cleaned DataFrame with valid float entries in the column.
    """
    def try_convert(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return None  # Mark as invalid

    # Apply conversion
    df[column_name] = df[column_name].apply(try_convert)

    # Drop rows with None (i.e., failed conversion)
    df_cleaned = df.dropna(subset=[column_name]).reset_index(drop=True)

    return df_cleaned

def convertDateColumn(df: pd.DataFrame, dateKey: str, timeKey: str):

    df['hour'] = int(str(df[timeKey]).split(':')[0].strip().replace(' ',''))
    df['month'] = int(str(df[dateKey]).split('-')[1].strip().replace(' ',''))
    df['year'] = int(str(df[dateKey]).split('-')[-1][:4].strip().replace(' ',''))

    # Drop original 'date' column (but not sorting the data)
    df = df.drop(columns=[dateKey, timeKey])
    return df

def getRandomForestModel(nEstimators: int = 100, randomState: int = 42, maxDepth: int = 20, minSampleLeaf:int = 10):
    """
    Function creates and returns **Random Forest Regression** model of type `sklearn.ensemble.RandomForestRegressor`.\n
    """
    rf = RandomForestRegressor(
        n_estimators=nEstimators, 
        random_state=randomState, 
        verbose=1, 
        max_depth=maxDepth, 
        min_samples_leaf=minSampleLeaf
        )
    return rf

def getPolynomialRegressionModel(learningRate: float = 0.1):
    """
    Function creates and returns **Polynomial Regression** model of type `sklearn.linear_model.Ridge`.\n
    """
    pr = Ridge(alpha=learningRate)
    return pr

def getXgBoostRegressionModel(nEstimators: int = 100, learningRate: float = 0.1, randomState:int = 42, maxDepth:int = 8):
    xgbModel = xgb.XGBRegressor(
        objective='reg:squarederror', 
        device='cuda',
        predictor='gpu_predictor',
        n_estimators=nEstimators, 
        learning_rate=learningRate, 
        random_state=randomState,
        max_depth=maxDepth,
        verbosity=1
        )
    return xgbModel

def getAnnRegressionModel(X_train_scaled: np.ndarray, optimizer:str, lossFunction:str):
    ANN = Sequential([
        Input(shape=(X_train_scaled.shape[1],)),  # Input layer (feature size must match X_train)
        Dense(64, activation='relu'),  # First hidden layer with ReLU activation
        Dense(32, activation='relu'),  # Second hidden layer
        Dense(1)  # Output layer (for regression, no activation function)
    ])
    # Compile the model
    ANN.compile(optimizer=optimizer, loss=lossFunction, metrics=['mae'])
    return ANN

def rfTrainer(
        filePath: str, 
        dateKey: str, 
        timeKey: str,
        targetKey: str, 
        testSize: float = 0.2, 
        randomState: int = 42,
        nEstimators: int = 100, 
        maxDepth: int = 5, 
        minSampleLeaf:int = 1,
        minSampleSplit:int = 2,
        nIterations:int = 20,
        numberOfValidations:int = 3
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
    
    param_dist = {
        'n_estimators': randint(nEstimators, 5*nEstimators),
        'max_depth': randint(maxDepth, 5*maxDepth),
        'min_samples_split': randint(minSampleSplit, 5*minSampleSplit),
        'min_samples_leaf': randint(minSampleLeaf, 5*minSampleLeaf),
        'max_features': ['auto', 'sqrt', 'log2']
    }

    df = pd.read_csv(filePath)        # Load the data from file

    df = df.dropna(subset=[dateKey, targetKey])
    df = convert_column_to_float_and_drop_invalid(df=df, column_name=targetKey)
    
    # Convert 'date' column to datetime and extract features
    df = convertDateColumn(df, dateKey, timeKey)

    # Split data into features (X) and target (y)
    X = df.drop(columns=[targetKey])  
    y = df[targetKey]

    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState)
    # print(f'\nX Train Data Types : {X_train.dtypes}')
    # print(f'Y Train Data Types : {y_train.dtypes}\n')

    model = getRandomForestModel(nEstimators, randomState, maxDepth, minSampleLeaf)

    # Setup RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=nIterations,  # Number of random combinations to try
        cv=numberOfValidations,       # 3-fold cross-validation
        scoring='r2',  # or 'neg_mean_squared_error', 'neg_mean_absolute_error'
        verbose=1,
        random_state=randomState,
        n_jobs=-1   # Use all available cores
    )

    print('\nBegin Random Forest Training....')
    # model.fit(X_train, y_train)
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    # Predict on test data
    y_pred = best_model.predict(X_test)

    # Compute evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print evaluation results
    print(f"Random Forest Results :\nMSE: {mse:.4f}\nMAE: {mae:.4f}\nR² Score: {r2:.4f}\n")
    print(f"Best Parameters for Random Forest : \n{json.dumps(random_search.best_params_, indent=4)}")

    # Plot Results of the Model
    plotRfResults(
        model=best_model, 
        X=X,
        y_test=y_test, 
        y_pred=y_pred, 
        )
    
def prTrainer(
        filePath: str, 
        dateKey: str,  
        timeKey: str,
        targetKey: str, 
        testSize: float = 0.2, 
        randomState: int = 42,
        learningRate: float = 0.1
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

    df = df.dropna(subset=[dateKey, targetKey])
    df = convert_column_to_float_and_drop_invalid(df=df, column_name=targetKey)

    # Convert 'date' column to datetime and extract features
    df = convertDateColumn(df, dateKey, timeKey)
    
    # Split data into features (X) and target (y)
    X = df.drop(columns=[targetKey])  
    y = df[targetKey]

    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState)
    # print(f'\nX Train Data Types : {X_train.dtypes}')
    # print(f'Y Train Data Types : {y_train.dtypes}\n')

    # Apply Polynomial Features Transformation (Degree 2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Scale the features (important for polynomial regression)
    X_train_poly = scaleData(X_train_poly)
    X_test_poly = scaleData(X_test_poly)

    model = getPolynomialRegressionModel(learningRate)

    # Train Polynomial Regression (Linear Regression on Polynomial Features)
    print('\nBegin Polynomial Regression Training....')
    model.fit(X_train_poly, y_train)

    # Predict on test data
    y_pred = model.predict(X_test_poly)

    # Compute evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print evaluation results
    print(f"Polynomial Regression Results :\nMSE: {mse:.4f}\nMAE: {mae:.4f}\nR² Score: {r2:.4f}\n")

    # Plot Results of the Model
    plotPrResults(
        y_test=y_test, 
        y_pred=y_pred, 
        )

def xgbTrainer(
        filePath: str, 
        dateKey: str,  
        timeKey: str,
        targetKey: str, 
        testSize: float = 0.2, 
        randomState: int = 42,
        nEstimators: int = 100, 
        learningRate: float = 0.1, 
        maxDepth:int = 8,
        crossValidations:int = 3
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
    
    param_grid = {
        'n_estimators': [round(nEstimators/2), nEstimators, 2*nEstimators],
        'max_depth': [3, maxDepth, 2*maxDepth],
        'learning_rate': [learningRate, learningRate/10, learningRate/100],
        'subsample': [0.8, 1.0],
        'tree_method': ['gpu_hist'],
        'device': ['cuda']
    }

    # Load the dataset
    df = pd.read_csv(filePath)

    df = df.dropna(subset=[dateKey, targetKey])
    df = convert_column_to_float_and_drop_invalid(df=df, column_name=targetKey)

    # Convert 'date' column to datetime and extract features
    df = convertDateColumn(df, dateKey, timeKey)

    # Split data into features (X) and target (y)
    X = df.drop(columns=[targetKey])  
    y = df[targetKey]

    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState)
    # print(f'\nX Train Data Types : {X_train.dtypes}')
    # print(f'Y Train Data Types : {y_train.dtypes}\n')

    model = getXgBoostRegressionModel(nEstimators, learningRate, randomState, maxDepth)

    # Define scoring metric (use R² or MAE or MSE)
    scorer = make_scorer(mean_squared_error, greater_is_better=False)  # Use 'neg_mean_squared_error' or 'r2' directly

    # Setup GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=crossValidations,  # 3-fold cross-validation
        scoring='r2',  # or 'neg_mean_absolute_error' / scorer
        verbose=3,
        n_jobs=-1  # Use all CPU cores
    )

    # Train XGBoost Regressor
    print(xgb.__version__)
    print(xgb.get_config()["use_rmm"])
    print('\nBegin XGBoost Model Training....\n')
    # model.fit(X_train, y_train)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Predict on test data
    y_pred = best_model.predict(X_test)

    # Compute evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print evaluation results
    print(f"XGBoost Model Results :\nMSE: {mse:.4f}\nMAE: {mae:.4f}\nR² Score: {r2:.4f}\n")
    print(f"XGBoost Model Results : \nBest Parameters : {grid_search.best_params_}\nBest R2 Score : {grid_search.best_score_}")
    print(f"Best Parameters for XGBoost : \n{json.dumps(grid_search.best_params_, indent=4)}")


    # Plot Results of the Model
    plotXgbResults(
        model=best_model,
        X=X,
        y_test=y_test, 
        y_pred=y_pred
        )
    
def AnnTrainer(
        filePath: str, 
        dateKey: str,  
        timeKey: str,
        targetKey: str, 
        testSize: float = 0.2, 
        randomState: int = 42,
        epochs: int = 50,
        batchSize: int = 32,
        nSplits:int = 5,
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

    df = df.dropna(subset=[dateKey, targetKey])
    df = convert_column_to_float_and_drop_invalid(df=df, column_name=targetKey)

    # Convert 'date' column to datetime and extract features
    df = convertDateColumn(df, dateKey, timeKey)

    # Split data into features (X) and target (y)
    X = df.drop(columns=[targetKey])  
    y = df[targetKey]

    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState)
    # print(f'\nX Train Data Types : {X_train.dtypes}')
    # print(f'Y Train Data Types : {y_train.dtypes}\n')

    # Standardize the features (important for ANN training)
    X_train_scaled = scaleData(X_train)
    X_test_scaled = scaleData(X_test)

    model = getAnnRegressionModel(X_train_scaled=X_train_scaled, optimizer=optimizer, lossFunction=lossFunction)

    # Wrap model for scikit-learn
    regressor = KerasRegressor(
        build_fn=model, 
        epochs=epochs, 
        batch_size=batchSize, 
        verbose=1,
        # random_state=randomState,
        # optimizer=optimizer,
        loss=lossFunction,
        metrics='mae',
        validation_split=0.1,
        shuffle=True,
        )

    print('\nBegin ANN Model Training....')

    # Cross-validation setup
    kfold = KFold(n_splits=nSplits, shuffle=True, random_state=randomState)

    # Perform cross-validation (use 'neg_mean_squared_error' for MSE)
    results = cross_val_score(regressor, X_test_scaled, y_test, cv=kfold, scoring='neg_mean_absolute_error')
    
    # Train the model
    # history = model.fit(
    #     X_train_scaled, 
    #     y_train, 
    #     epochs=epochs, 
    #     batch_size=batchSize, 
    #     validation_data=(X_test_scaled, y_test), 
    #     verbose=1
    #     )

    # Predict on test data
    y_pred = model.predict(X_test_scaled).flatten()

    # Compute evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print evaluation results
    print(f"ANN Model Results :\nMSE: {mse:.4f}\nMAE: {mae:.4f}\nR² Score: {r2:.4f}\n")
    print(f"Mean for each fold: {results}\nMean MAE: {results.mean()}")

    # Plot Results of the Model
    # plotAnnResults(
    #     y_test=y_test,
    #     y_pred=y_pred,
    #     history=history.history
    # )