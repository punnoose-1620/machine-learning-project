import json
import requests
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
        print()
        if data is not None:
            write_json_to_file(f'./electricityPrices/{dateAsString}.json', data=data)
            print(f"Swedish Electricity Price Data for {dateAsString} written to file....\n")

# getElectricityExportedToGermany()
# getElectricityPrices()