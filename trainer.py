from constants import *
from modelFunctions import *

date_key = 'Date'
time_key = 'Time'
target_key = 'Price'

def readCsv(filePath: str, targetColumn: str):
    data = pd.read_csv(filePath)
    
    # Drop rows with missing target values
    data = data.dropna(subset=[targetColumn])
    return data

data = readCsv(finalMergedFile, target_key)

# Specify Parameters

N_ESTIMATORS = 250
RANDOM_STATE = 42
LEARNING_RATE = 0.01
TEST_SIZE = 0.3
EPOCHS = 100
BATCH_SIZE = 32
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'mse'
MAX_TREE_DEPTH = 10
LEAF_SIZE = 5
REGRESSION_DEGREE = 3

# Create Models

rfModel = getRandomForestModel(nEstimators=N_ESTIMATORS, randomState=RANDOM_STATE, maxDepth=MAX_TREE_DEPTH, minSampleLeaf=LEAF_SIZE)
prModel = getPolynomialRegressionModel(learningRate=LEARNING_RATE)
xgbModel = getXgBoostRegressionModel(nEstimators=N_ESTIMATORS, learningRate=LEARNING_RATE, randomState=RANDOM_STATE, maxDepth=MAX_TREE_DEPTH)

# invokations for training

rfTrainer(
    model=rfModel, 
    filePath=finalMergedFile, 
    dateKey=date_key, 
    timeKey=time_key,
    targetKey=target_key, 
    testSize=TEST_SIZE, 
    randomState=RANDOM_STATE
    )
# print('\n\n')
prTrainer(
    model=prModel, 
    filePath=finalMergedFile, 
    dateKey=date_key, 
    timeKey=time_key, 
    targetKey=target_key, 
    testSize=TEST_SIZE, 
    randomState=RANDOM_STATE
    )
# print('\n\n')
xgbTrainer(
    model=xgbModel, 
    filePath=finalMergedFile, 
    dateKey=date_key, 
    timeKey=time_key, 
    targetKey=target_key, 
    testSize=TEST_SIZE, 
    randomState=RANDOM_STATE
    )
# print('\n\n')
AnnTrainer(
    filePath=finalMergedFile, 
    dateKey=date_key, 
    timeKey=time_key, 
    targetKey=target_key, 
    testSize=TEST_SIZE, 
    randomState=RANDOM_STATE, 
    epochs=EPOCHS, 
    batchSize=BATCH_SIZE, 
    optimizer=OPTIMIZER, 
    lossFunction=LOSS_FUNCTION)