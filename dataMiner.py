import json
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

def write_json_to_file(file_path, data):
    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)
        print(f"Data successfully written to {file_path}")
    except Exception as e:
        print(f"Error writing JSON: {e}")

def read_json_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        print(f"Data successfully read from {file_path}")
        return data
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return None

# Energy values are represented in Giga Watts (GW)
def getYearBasedElectricityExportedToGermany(year: str, countryName: str):
    url = f"https://www.energy-charts.info/charts/power/data/de/year_tcs_saldo_{year}.json"
    try:
        response = requests.get(url=url)
        byteData = response.content
        jsonData = json.loads(byteData) # Returns array of objects
        print(f"Fetch Electricity Prices response : {type(response.content)} : {type(jsonData)} : length({len(jsonData)})")
        for data in jsonData:
            countryName = data['name'][0]['en']
            if countryName==countryName:
                return data
        print("Country Name not found in dataset")
    except Exception as e:
        print(f"Fetch Electricity Prices error : {e}")
    return None

def getWeekBasedElectricityExportedToGermany(year: str, week: str, countryName: str):
    url = f"https://www.energy-charts.info/charts/power/data/de/year_tcs_saldo_{year}_{week}.json"
    try:
        response = requests.get(url=url)
        byteData = response.content
        jsonData = json.loads(byteData) # Returns array of objects
        print(f"Fetch Electricity Prices response : {type(response.content)} : {type(jsonData)} : length({len(jsonData)})")
        for data in jsonData:
            countryName = data['name'][0]['en']
            if countryName==countryName:
                return data
        print("Country Name not found in dataset")
    except Exception as e:
        print(f"Fetch Electricity Prices error : {e}")
    return None

def getElectricityExportedToGermany():
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    for year in years:
        data = getYearBasedElectricityExportedToGermany(year=year, countryName='Sweden')
        if data is not None:
            write_json_to_file(f'./electricityExportAmounts/{year}.json', data=data)
            print(f"Electricity Export Data for {year} written to file....\n")

def getDateBasedElectricityPrice(yyyymmdd: str):
    # Get BZN|SE3
    url = f"https://thingler.io/day-ahead?date={yyyymmdd}&bz=BZN|SE1,BZN|SE2,BZN|SE3,BZN|SE4"
    try:
        response = requests.get(url)
        data = json.loads(response.content)
        print(f"Get Electricity response : {data.keys()} : {'BZN|SE3' in data.keys()}")
        if('BZN|SE3' in data.keys()):
            return data['BZN|SE3']
    except Exception as e:
        print(f"Get Electricity error {e}")
    return None

def getElectricityPrices():
    today = datetime.today().date()
    start_date = datetime(2025, 1, 1).date()
    current_date = start_date
    while current_date <= today:
        current_date += timedelta(days=1)
        dateAsString = f'{current_date.strftime("%Y-%m-%d")}'
        data = getDateBasedElectricityPrice(dateAsString)
        if data is not None:
            write_json_to_file(f'./electricityPrices/{dateAsString}.json', data=data)
            print(f"Swedish Electricity Price Data for {dateAsString} written to file....\n")

def getParametersList():
    url = "https://opendata-download-metobs.smhi.se/api/version/1.0/parameter.json"
    try:
        response = requests.get(url)
        data = json.loads(response.content)
        dataKeys = data.keys()
        # print(f"Get Parameters List response : {json.dumps(data, indent=4)}")
        # print(f"\nGet Parameters List response : {json.dumps(data['key'], indent=4)}")
        # print(f"Get Parameters List response : {json.dumps(data['updated'], indent=4)}")
        # print(f"Get Parameters List response : {json.dumps(data['title'], indent=4)}")
        # print(f"Get Parameters List response : {json.dumps(data['summary'], indent=4)}")
        # print(f"Get Parameters List response : {json.dumps(data['link'], indent=4)}")
        # print(f"Get Parameters List response : {json.dumps(data['resource'], indent=4)}")
        print(f"Get Parameters List response : {data.keys()}")
        if('resource' in dataKeys):
            resources = data['resource']
            returnVal = []
            for item in resources:
                # print(f'{item['key']} : {item['title']} : {item['unit']}\n\n')
                returnVal.append({'id': item['key'], 'unit': item['unit'], 'title': item['title']})
            return returnVal
    except Exception as e:
        print(f"Get Parameters List error {e}")
    return None

def fetch_smhi_weather(station_id, parameter_id, period="latest-day"):
    """
    Fetches historical weather data from SMHI for a given station and parameter.

    Args:
        station_id (int): The ID of the weather station.
        parameter_id (int): The parameter ID (e.g., temperature, precipitation).
        period (str): The time period ("latest-hour", "latest-day", "latest-month", etc.).
                      Default is "latest-day".

    Returns:
        dict: Weather data from SMHI.
    """
    base_url = f"https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/{parameter_id}/station/{station_id}/period/{period}/data.json"
    
    try:
        response = requests.get(base_url)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def getHydroParams():
    contributionFactor = 0.4
    start_year = 1800
    # Nearest Station to Hydro Plants affecting SE3 prices along with Keys for Percipitation on different stations
    stats_and_params = {114140: [23, 14, 5, 7]} 

def getNuclearParams():
    contributionFactor = 0.3
    start_year = 1800
    # Nearest Station to Nuclear Plants affecting SE3 prices along with Keys for Percipitation on different stations
    stats_and_params = {}

def getBioEnergyParams():
    contributionFactor = 0.08
    start_year = 1800
    # Nearest Station to Bio Energy Plants affecting SE3 prices along with Keys for Percipitation on different stations
    stats_and_params = {}

def getSolarParams():
    contributionFactor = 0.01
    start_year : 1983
    # Nearest Station to Solar Plants affecting SE3 prices along with Keys for Percipitation on different stations
    stats_and_params = {93235: [], 86655: []}

# Dalarna : Hydro [114140] : 40%
# Uppsala, Haland : Nuclear [108640, 72160] : 30%
# Stockholm, Norrkoping : BioEnergy [98200, 98100, 86360] : 8%
# Sodermanland, Ostergotland, Vastra Gotaland : Solar [97150, 85180, 84390] : 1%

#Stations and Parameters affecting SE3 Prices {station: parameter}
stats_and_params = {
    1: 1,
    4: 4,
    1: 6,
    1: 7,
    1: 910,
    1: 22
}

# getElectricityExportedToGermany()
# getElectricityPrices()
paramData = getParametersList()
print(f"Parameters : {json.dumps(paramData, indent=4)}")