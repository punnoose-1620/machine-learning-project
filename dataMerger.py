from dataMiner import *

def mergeSwedishWeatherData():
    hydroData = getHydroParams()
    solarData = getSolarParams()
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
    first_data = getTotalSwedishElectricityExport(datetime(year=startYear, month=startMonth, day=startDay))
    second_data = getElectricityPrices(datetime(year=startYear, month=startMonth, day=startDay), False)
    finalData = {}
    firstKeys = list(first_data.keys())
    secondKeys = list(second_data.keys())
    if(first_data is not None and second_data is not None):
        for key in firstKeys:
            if key in secondKeys:
                previousPrice = 0
                if key!=firstKeys[0]:
                    index = firstKeys.index(key)
                    tempKey = firstKeys[index-1]
                    if tempKey in secondKeys:
                        previousPrice = second_data[tempKey]
                finalData[key] = {
                'exported' : first_data[key],
                'price' : second_data[key],
                'previousPrice' : previousPrice
            }
    write_json_to_file('./mergedPricesTest.json', finalData)
    return finalData

mergeElectricityData(2025, 2, 1)