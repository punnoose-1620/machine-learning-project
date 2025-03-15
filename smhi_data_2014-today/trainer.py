from constants import *
from modelFunctions import *

date_key = 'Date'
target_key = ''

def readCsv(filePath: str, targetColumn: str):
    data = pd.read_csv(filePath)
    
    # Drop rows with missing target values
    data = data.dropna(subset=[targetColumn])
    return data

data = readCsv(finalMergedFile, target_key)

# invokations for training
