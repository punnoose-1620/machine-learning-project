from dataMiner import *

def mergeSwedishWeatherData(startYear: int, startMonth: int, startDay: int):
    hydroData = getHydroParams(startYear, startMonth, startDay)
    solarData = getSolarParams(startYear, startMonth, startDay)
    # Weather Format : {datehour : { param : value } }
    hydroDates = hydroData.keys()
    solarDates = solarData.keys()
    finalDates = []     # List of dates common to hydro and solar
    finalData = []
    for item in hydroDates:
        if item in solarDates:
            finalDates.append(item)
    for item in finalDates:
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
    write_json_to_file('./mergedPricesTest.json', mergedData)
    return mergedData

# mergeElectricityData(2025, 2, 1)
mergeSwedishWeatherData(2024 ,2, 1)