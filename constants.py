# Below this line is SE3
a3 = (61.532329, 11.963278)
b3 = (61.156689, 17.719862) 

# Folder Paths
electricityBillsFolder = './electricity'
plotsFolder = './GraphPlots'
mergedPlots = './GraphPlots/MergedPlots'
existingWeatherFolder = './smhi_data'
mergedWeatherFolder = './smhi_data_2014-today'
testJsonWeatherFile = './mergedWeatherTest.json'
testJsonElectricityFile = './mergedPricesTest.json'

# Swedish Regions of Electricity Grids
regions = ['SE1','SE2','SE3','SE4']

# List of valuable params from mined weather data
final_params = ['Solskenstid', 'Lufttemperatur', 'Nederbördsmängd', 'Snödjup']