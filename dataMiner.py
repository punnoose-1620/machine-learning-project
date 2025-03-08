import json
import time
import requests
import pandas as pd
from constants import *
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
def getTotalSwedishElectricityExport(startDate: datetime):
    '''
    Gets the total hourly Electricity Export from Sweden from the given start datetime to today.\n
    Returns amount in Giga Watts\n
    Expected return format : { date1_hour1 : exportValue1, date1_hour2 : exportValue1 }
    '''
    countryCode = 'se'
    startEpochTime = str(time.mktime(startDate.timetuple())).split('.')[0]
    endEpochTime = str(time.mktime(datetime.now().timetuple())).split('.')[0]
    params = {
        'country': countryCode,
        'start': startEpochTime,
        'end': endEpochTime
    }
    url = f'https://api.energy-charts.info/cbet?country={countryCode}&start={startEpochTime}&end={endEpochTime}'
    print(f"Get Electricity Export Params : {params}")
    try:
        response = requests.get(url)
        data = json.loads(response.content)
        print(f"Get Electricity Export response : {data.keys()}\n")
        unixSeconds = data['unix_seconds']
        countries = data['countries']
        deprecated = data['deprecated']
        if deprecated==True:
            return None
        exportSums=[]
        for country in data['countries']:
          if country['name'] == 'sum':
             exportSums = country['data']
             break
        #print("Export Sums found jini",exportSums)
                
        print(f"Export Sums({len(exportSums)})")

        finalData = {}
        for i in range(len(unixSeconds)):
            entryDate = datetime.fromtimestamp(unixSeconds[i])
            exportValue = exportSums[i]
            # print(f"{entryDate} : {exportValue}")
            if entryDate in finalData:
                print("Test Error")
            finalData[f'{entryDate}'] = exportValue
        print(f"Total Swedish Export Final Data : {json.dumps(finalData, indent=4)}")
        return finalData
    except Exception as e:
        print(f"Get Electricity Export error {e}")
    return None

def getDateBasedElectricityPrice(yyyymmdd: str):
    # Get BZN|SE3
    url = f"https://thingler.io/day-ahead?date={yyyymmdd}&bz=BZN|SE1,BZN|SE2,BZN|SE3,BZN|SE4"
    try:
        response = requests.get(url)
        data = json.loads(response.content)
        # print(f"Get Electricity response : {data.keys()}")
        if('BZN|SE3' in data.keys()):
            return data['BZN|SE3']
    except Exception as e:
        print(f"Get Electricity error {e}")
    return None

def getElectricityPrices(startDate: datetime, write: bool):
    '''
    Gets hourly electricity prices from given start date to current time.\n
    Expected output format is { date0_hour0 : price_value0, date0_hour1 : price_value1 }
    '''
    today = datetime.today().date()
    start_date = startDate.date()
    current_date = start_date
    finalData = {}
    while current_date <= today:
        current_date += timedelta(days=1)
        dateAsString = f'{current_date.strftime("%Y-%m-%d")}'
        data = getDateBasedElectricityPrice(dateAsString)
        if data is not None:
            for item in data:
                key = item['time']
                value = item['price']
                refKeys = finalData.keys()
                if key not in refKeys:
                    finalData[key] = value
            if write==True:
                write_json_to_file(f'./electricityPrices/{dateAsString}.json', data=data)
                print(f"Swedish Electricity Price Data for {dateAsString} written to file....\n")
    print(f"Electricity Prices Final Data : {json.dumps(finalData, indent=4)}")
    return finalData

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

# Get SMHI Weather Data
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
        print(f"\nError fetching Specific SMHI Weather data: {e}\n")
        return None

def getHydroParams(startYear: int, startMonth: int, startDay: int):
    contributionFactor = 0.4
    start_year = 1800
    # Nearest Station to Hydro Plants affecting SE3 prices along with Keys for Percipitation on different stations
    # stats_and_params = {114140: [23, 14, 5, 7], 2396: [], 1906: []} 
    stats = [114140]
    params = [7]
    # For each station, get all required param values
    # For each date in both stations, take union of params, take average for common params
    # Expected Output : { date : { hydro_param0 : value0, hydro_param1 : value1 } }
    final_data = {}
    for s in stats:
        for p in params:
            data = fetch_smhi_weather(s,p)
            if data is not None:
                print(f"Hydro data station({s}) param({p}) response from : {datetime.fromtimestamp(0)+timedelta(seconds=float(data['period']['from'])/1000)} : {datetime.fromtimestamp(0)+timedelta(seconds=float(data['period']['to'])/1000)}")
                print(f"Params : {data.keys()}")
                # print(f'Values : {json.dumps(data['value'], indent=4)}')
                for item in data['value']:
                    # print(f"Value test : {item} : {type(item)}")
                    dateTime = float(item['date'])
                    dateKey = f'{datetime.fromtimestamp(float(dateTime/1000))}'
                    value = item['value']
                    if dateKey in final_data.keys():
                        if p in final_data[dateKey].keys():
                            final_data[dateKey] = (final_data[dateKey][p]+value)/2
                        else:
                            final_data[dateKey][p] = value
                    else:
                        final_data[dateKey] = {p: value}
    print(f"Get Hydro Params Final Data : {json.dumps(final_data, indent=4)}")

def getNuclearParams():
    contributionFactor = 0.3
    start_year = 1800
    # Nearest Station to Nuclear Plants affecting SE3 prices along with Keys for Percipitation on different stations
    stats_and_params = {}
    # For each station, get all required param values
    # For each date in both stations, take union of params, take average for common params
    # Expected Output : { date : { nuclear_param0 : value0, nuclear_param1 : value1 } }

def getBioEnergyParams():
    contributionFactor = 0.08
    start_year = 1800
    # Nearest Station to Bio Energy Plants affecting SE3 prices along with Keys for Percipitation on different stations
    stats_and_params = {}
    # For each station, get all required param values
    # For each date in both stations, take union of params, take average for common params
    # Expected Output : { date : { bio_param0 : value0, bio_param1 : value1 } }

def getSolarParams(startYear: int, startMonth: int, startDay: int):
    contributionFactor = 0.01
    start_year : 1983
    # Nearest Station to Solar Plants affecting SE3 prices along with Keys for Percipitation on different stations
    stats_and_params = {93235: [], 86655: []}
    stats = [93235, 86655]
    params = [5, 8, 10]
    # For each station, get all required param values
    # For each date in both stations, take union of params, take average for common params
    # Expected Output : { date : { solar_param0 : value0, solar_param1 : value1 } }
    final_data = {}
    for s in stats:
        for p in params:
            data = fetch_smhi_weather(s,p)
            print(f"Data : {data}")
            if data is not None:
                print(f"Solar data station({s}) param({p}) response from : {datetime.fromtimestamp(0)+timedelta(seconds=float(data['period']['from'])/1000)} : {datetime.fromtimestamp(0)+timedelta(seconds=float(data['period']['to'])/1000)}")
                print(f"Params : {data.keys()}")
                # print(f'Values : {json.dumps(data['value'], indent=4)}')
                for item in data['value']:
                    # print(f"Value test : {item} : {type(item)}")
                    dateTime = float(item['date'])
                    dateKey = f'{datetime.fromtimestamp(float(dateTime/1000))}'
                    value = int(item['value'])
                    if dateKey in final_data.keys():
                        if p in final_data[dateKey].keys():
                            final_data[dateKey] = (final_data[dateKey][p]+value)/2
                        else:
                            final_data[dateKey][p] = value
                    else:
                        final_data[dateKey] = {p: value}
    print(f"Get Solar Params Final Data : {json.dumps(final_data, indent=4)}")

# Dalarna : Hydro s[114140] p[2, 5, 8, 10] : 40%
# Uppsala, Haland : Nuclear [108640, 72160] : 30%
# Stockholm, Norrkoping : BioEnergy s[98200, 98100, 86360] p[]: 8%
# Sodermanland, Ostergotland, Vastra Gotaland : Solar s[97150, 85180, 84390] p[28, 30, 32, 10] : 1%

# getElectricityExportedToGermany()
# getElectricityPrices()
# paramData = getParametersList()
# print(f"Parameters : {json.dumps(paramData, indent=4)}")

# getSolarParams(2025, 2, 1)
# startDate = datetime(2025, 2, 25, 0, 0)
# getTotalSwedishElectricityExport(startDate)