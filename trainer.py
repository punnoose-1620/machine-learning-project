import warnings
from constants import *
from modelFunctions import *

date_key = 'timestamp'
time_key = ''
target_key = 'price'

def readCsv(filePath: str, targetColumn: str):
    data = pd.read_csv(filePath)
    
    # Drop rows with missing target values
    data = data.dropna(subset=[targetColumn])
    return data

# data = readCsv(finalMergedFile, target_key)

# Specify Parameters

N_ESTIMATORS = 300
RANDOM_STATE = 42
LEARNING_RATE = 0.1
TEST_SIZE = 0.3
EPOCHS = 100
BATCH_SIZE = 64
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'mse'
MAX_TREE_DEPTH = 5
LEAF_SIZE = 10
SAMPLE_SPLIT_SIZE = 10
REGRESSION_DEGREE = 3
N_ITERATIONS = 50
CROSS_VALIDATIONS_COUNT = 25

warnings.filterwarnings("ignore")

# invokations for training

# rfTrainer(
#     filePath=finalMergedFile, 
#     dateKey=date_key, 
#     timeKey=time_key,
#     targetKey=target_key, 
#     testSize=TEST_SIZE, 
#     randomState=RANDOM_STATE,
#     nEstimators=N_ESTIMATORS,
#     maxDepth=MAX_TREE_DEPTH,
#     minSampleLeaf=LEAF_SIZE, 
#     minSampleSplit=SAMPLE_SPLIT_SIZE,
#     nIterations=N_ITERATIONS,
#     numberOfValidations=CROSS_VALIDATIONS_COUNT
#     )

# prTrainer(
#     filePath=finalMergedFile, 
#     dateKey=date_key, 
#     timeKey=time_key, 
#     targetKey=target_key, 
#     testSize=TEST_SIZE, 
#     randomState=RANDOM_STATE,
#     learningRate=LEARNING_RATE
#     )

# xgbTrainer( 
#     filePath=finalMergedFile, 
#     dateKey=date_key, 
#     timeKey=time_key, 
#     targetKey=target_key, 
#     testSize=TEST_SIZE, 
#     randomState=RANDOM_STATE,
#     nEstimators=N_ESTIMATORS,
#     learningRate=LEARNING_RATE,
#     maxDepth=MAX_TREE_DEPTH,
#     crossValidations=CROSS_VALIDATIONS_COUNT
#     )

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
    lossFunction=LOSS_FUNCTION
    )