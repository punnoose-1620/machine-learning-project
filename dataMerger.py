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

def mergeElectricityData():
    return []