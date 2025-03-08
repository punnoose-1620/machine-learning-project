import os
import pandas as pd
from tqdm import tqdm
from dataMiner import *
from constants import *

franCount = 0

def mergeSwedishWeatherData(startYear: int, startMonth: int, startDay: int):
    hydroData = getHydroParams(startYear, startMonth, startDay)
    solarData = getSolarParams(startYear, startMonth, startDay)
    # Weather Format : {datehour : { param : value } }
    if hydroData is not None and solarData is not None:
        hydroDates = hydroData.keys()
        solarDates = solarData.keys()
        finalDates = []     # List of dates common to hydro and solar
        finalData = []
        for item in hydroDates:             # Get list of common dates
            if item in solarDates:
                finalDates.append(item)
        for item in finalDates:             # Get values for list of common dates
            hParamKeys = hydroData[item].keys()
            sParamKeys = solarData[item].keys()
            tempItem = {}
            for key in hParamKeys:
                tempItem[key] = hydroData[item][key]
            for key in sParamKeys:
                tempItem[key] = solarData[item][key]
            if tempItem!={}:
                finalData.append(tempItem)
        print(f"Merged Weather Data : {finalData}")
        write_json_to_file(testJsonWeatherFile, finalData)
        return finalData

def mergeElectricityData(startYear: int, startMonth: int, startDay: int):
    exports_data = getTotalSwedishElectricityExport(datetime(year=startYear, month=startMonth, day=startDay))
    prices_data = getElectricityPrices(datetime(year=startYear, month=startMonth, day=startDay), False)
    mergedData = {}
    exportsKeys = list(exports_data.keys())     # List of all date-hour values used as keys in exports data
    pricesKeys = list(prices_data.keys())       # List of all date-hour values used as keys in prices data
    if(exports_data is not None and prices_data is not None):
        for key in exportsKeys:
            if key in pricesKeys:
                previousHourPrice = 0
                if key!=exportsKeys[0]:
                    index = exportsKeys.index(key)
                    tempKey = exportsKeys[index-1]
                    if tempKey in pricesKeys:
                        previousHourPrice = prices_data[tempKey]
                mergedData[key] = {
                'exported' : exports_data[key],
                'price' : prices_data[key],
                'previousPrice' : previousHourPrice
            }
    write_json_to_file(testJsonElectricityFile, mergedData)
    return mergedData

def readCsvFromHeaders(filepath, target_header):
    global franCount
    emptyFlag = False
    skip_rows = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if 'Från' in line:
                franCount = franCount+1
            if target_header in line:  # Find the line index where the header is
                skip_rows = i  # Set the number of rows to skip
                break
        else:
            emptyFlag = True
            # raise ValueError(f"\nHeader '{target_header}' not found in {filepath}\n")

    # Read the CSV while skipping the detected rows
    if(emptyFlag==False):
        df = pd.read_csv(filepath, skiprows=skip_rows, delimiter='\t')    
    else:
        df = pd.DataFrame() 
    return df

def processWeatherFileData(df: pd.DataFrame, columnsList: list, param: str):
    dateTimeKey = ''
    processedData = {}
    global final_params
    for index, row in df.iterrows():
        tempData = {}
        processedKeys = str(list(row.to_dict().keys())[0]).split(';')                   # Keys of each entry
        processedValues = str(row.to_dict()[list(row.to_dict().keys())[0]]).split(';')  # Values of each entry
        valueTitle = final_params[0]
        for x in range(len(processedKeys)):
            tempKey = f'{str(processedKeys[x])[:-2]}00'
            if tempKey in final_params:
                valueTitle = tempKey
            if x<len(processedValues):
                tempData[tempKey] = processedValues[x]
            else:
                tempData[tempKey] = ''
        value = tempData[valueTitle]
        # value = {param : }
        if('Från' in columnsList[0]):
            dateTimeKey = f'{processedValues[0]}'
        else:
            dateTimeKey = f'{processedValues[0]} {processedValues[1]}'
        dateTimeKey = f'{dateTimeKey[:-2]}00'
        processedData[dateTimeKey] = value
    return processedData

def filterExistingWeatherData():
    """
    Reads already fetched files from `smhi_data` folder.\n
    Merges data from all folders and all years into a single `dict`\n
    Writes merged data to `mergedWeatherTest.json` for debugging.

    """
    finalData = {}
    for subdir, _, files in os.walk(existingWeatherFolder):
        parent_folder = os.path.basename(subdir)  # Get the parent folder name
        parameter = parent_folder.split('_')[-1]
        columnsList = []
        for file in tqdm(files, desc=f"Parsing Existing Weather Files at parameter {parameter} folder"):
            if file.endswith(".csv"):
                file_path = os.path.join(subdir, file)
                try:
                    df = readCsvFromHeaders(filepath=file_path, target_header='Datum')
                except pd.errors.EmptyDataError:
                    print(f"Warning: {file_path} is empty. Returning an empty DataFrame.")
                    df = pd.DataFrame() 
                finally:
                    columns = df.columns
                    filteredList = str(columns).replace("Index(['",'').replace("'], dtype='object')",'').split(';')
                    for item in filteredList:
                        if(item not in columnsList and item.strip()!=''):
                            columnsList.append(item)
                    processedData = processWeatherFileData(df, filteredList, parameter)
                    for key in processedData.keys():
                        value = 0.0
                        if(processedData[key]!=''):
                            value = float(processedData[key])
                        if(isinstance(value, str)):
                            print(f'processed data test : ( {key} : {value} )')
                        refKeys = finalData.keys()
                        if key in refKeys:
                            if parameter in finalData[key].keys():
                                finalData[key][parameter] = (finalData[key][parameter]+value)/2
                            else:
                                finalData[key][parameter] = value
                        else:
                            finalData[key] = {parameter: value}
    write_json_to_file(testJsonWeatherFile, finalData)
    print(f'\nFrån count : {franCount}')
    print(f'File count : {len(files)}')
    print(f"\n All type of columns : {columnsList}")
    print(f"\nFinal Data Keys : {finalData.keys}\nFinal Data : {json.dumps(finalData,  indent=4)}")

# mergeElectricityData(2025, 2, 1)
# mergeSwedishWeatherData(2024 ,2, 1)
filterExistingWeatherData()