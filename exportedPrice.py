
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt

def fetch_data(api_url):
    # Send a GET request to the API
    response = requests.get(api_url)
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return None

def process_and_visualize_data(data):
    # Convert unix_seconds to datetime
    timestamps = pd.to_datetime(data['unix_seconds'], unit='s')

    # Extract 'sum' country data
    sum_data = None
    for country in data['countries']:
        if country['name'] == 'sum':
            sum_data = country['data']
            break
    
    if sum_data is None:
        print("No 'sum' data found")
        return

    # Create a DataFrame
    df = pd.DataFrame({
        'date': timestamps,
        'sum': sum_data
    })
    
    # Set date as index
    df.set_index('date', inplace=True)
    
    # Convert sum to positive values
    df['sum'] = df['sum'].abs()

    # Extract unique years
    years = df.index.year.unique()

    # Aggregate data by month and year
    monthly_aggregated = df.groupby([df.index.year, df.index.month]).sum().unstack(level=0)

    # Plot the positive sum exported electricity over time for each year
    plt.figure(figsize=(12, 8))
    for year in years:
        plt.plot(monthly_aggregated.index, monthly_aggregated['sum'][year], marker='o', linestyle='-', label=f'{year}')

    plt.title('Monthly Positive Exported Electricity Over Time by Year')
    plt.xlabel('Month')
    plt.ylabel('Exported Electricity')
    plt.grid(True)
    plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend()
    plt.show()

# Define the API endpoint
api_url = 'https://api.energy-charts.info/cbet?country=se&start=2015-01-01&end=2024-12-31'

# Fetch and process the data
data = fetch_data(api_url)
if data:
    process_and_visualize_data(data)