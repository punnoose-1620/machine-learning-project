import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None

electricityBillsFolder = './electricity'
plotsFolder = './GraphPlots'
mergedPlots = './GraphPlots/MergedPlots'

regions = ['SE1','SE2','SE3','SE4']

def plotHourlyData(df, df_name, time_column, value_column):
    try:
        regionName = str(str(df_name).split('_')[0])
        fileName = str(str(df_name).split('_')[-1])
        targetFile = os.path.join(plotsFolder+'/'+regionName, fileName)
        # Extract the start time from the 'MTU (CET/CEST)' column
        df['Timestamp'] = pd.to_datetime(df[time_column].str.split(' - ').str[0], format='%d.%m.%Y %H:%M')

        # Sort by time (optional but ensures correct order)
        df = df.sort_values(by='Timestamp')

        # Plot the data
        plt.figure(figsize=(12, 6))
        plt.plot(df['Timestamp'], df[value_column], marker='o', linestyle='-', color='b', label=value_column)

        # Formatting the plot
        plt.xlabel("Time (Hourly)")
        plt.ylabel(value_column)
        plt.title(f"{fileName.replace('.png','')}")
        plt.xticks(rotation=45)  # Rotate labels for better readability
        plt.legend()
        plt.grid(True)

        # Save the plot as a high-resolution PNG file
        plt.savefig(targetFile, dpi=1200, bbox_inches='tight')

        # Show the plot
        # plt.show()
        plt.close()
    
    except Exception as e:
        print(f"Plot Hourly Rates Error: {e}")

def readCsv(filePath: str):
    try:
        # Read CSV into a DataFrame
        df = pd.read_csv(filePath)
        
        # Print Data
        print(f"\nTitles : {df.columns.tolist()}")
        print(df.head())  # Print first few rows
        print(";" * 50)  # Separator for readability
        return df
    
    except Exception as e:
        print(f"Read CSV Error: {e}")

def mergePlots(sourceFolder, destinationFileName, horizontalFlag):
    images = []
    for file in sorted(os.listdir(sourceFolder)):
        if file.endswith(".png"):
            img = Image.open(os.path.join(sourceFolder, file))
            images.append(img)
    
    # Calculate total size (assuming you want a vertical merge)
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)
    if horizontalFlag==True:
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)

    # Create a blank canvas with the total size
    merged_image = Image.new('RGB', (max_width, total_height))
    if horizontalFlag==True:
        merged_image = Image.new('RGB', (total_width, max_height))

    y_offset = 0
    x_offset = 0
    for img in images:
        if horizontalFlag==True:
            merged_image.paste(img, (x_offset, 0))
            x_offset += img.width
        else:
            merged_image.paste(img, (0, y_offset))
            y_offset += img.height
    
    # Save the merged image
    merged_image.save(destinationFileName, format='PNG')

# files = [f for f in os.listdir(electricityBillsFolder) if f.endswith('.csv')]
# for file in files:
#     filePath = os.path.join(electricityBillsFolder, file)
#     df = readCsv(filePath)
#     plotHourlyData(df, file.replace('_entsoe.csv','.png'), 'MTU (CET/CEST)', 'Day-ahead Price [EUR/MWh]')

# Merge All Annual Electricity Data in a Region
# for region in regions:
#     regionFolder = os.path.join(plotsFolder,region)
#     dest = os.path.join(mergedPlots,region+'.png')
#     print(f"Loading Regional Plots for {region} to {dest}")
#     mergePlots(regionFolder, dest, False)

# Merge All Data from All regions
finalDestination = os.path.join(mergedPlots,'ElectricityGraphs.png')
mergePlots(mergedPlots,finalDestination, True)
