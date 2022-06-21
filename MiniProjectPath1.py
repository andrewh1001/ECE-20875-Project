from matplotlib import gridspec
import pandas
import matplotlib.pyplot as plt
from datetime import datetime 
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
import numpy as np
''' 
The following is the starting code for path1 for data reading to make your first step easier.
'dataset_1' is the clean data for path1.
'''
dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
dataset_1['Brooklyn Bridge']      = pandas.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
dataset_1['Manhattan Bridge']     = pandas.to_numeric(dataset_1['Manhattan Bridge'].replace(',','', regex=True))
dataset_1['Queensboro Bridge']    = pandas.to_numeric(dataset_1['Queensboro Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
#dataset_1['Date'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y')) 
# print(dataset_1.to_string()) #This line will print out your data

date = dataset_1["Date"] + '-2016'
date = pandas.to_datetime(date)
highTemp = dataset_1['High Temp']
lowTemp = dataset_1['Low Temp']
precipt = dataset_1['Precipitation']
brookBrd = dataset_1['Brooklyn Bridge']
manBrd = dataset_1['Manhattan Bridge']
queenBrd = dataset_1['Queensboro Bridge']
williamsBrd = dataset_1['Williamsburg Bridge']

plt.figure(1)
grid = plt.GridSpec(2,2)

#Temperature
plt.subplot(grid[0,:])
plt.plot(date,highTemp, label = 'Highest Temperature', color = 'r')
plt.plot(date,lowTemp, label = 'Lowest Temperature', color = 'b')
ax = plt.gca()
plt.ylim([20, 100])
ax.set_yticks([20, 40, 60, 80, 100])
plt.xticks(rotation=90)
locator = AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(AutoDateFormatter(locator) )
plt.xlabel('Time')
plt.ylabel('Temperature (Fahrenheit)')
ax.set_title('Highest and Lowest Temperature in One Day')
plt.grid()
plt.legend(fontsize = 'small')

#Precipitation
plt.subplot(grid[1,:])
plt.plot(date,precipt)
ax = plt.gca()
plt.xticks(rotation=90)
locator = AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(AutoDateFormatter(locator) )
plt.xlabel('Time')
plt.ylabel('Precipitation (in)')
ax.set_title('Rain Drop Height')
plt.grid()

#Subplot Config
plt.suptitle('Temperature and Precipitation', fontsize  = 'xx-large')
plt.tight_layout()

#Bridges Bike Usage Graph
plt.figure(2)
grid = plt.GridSpec(2,2)

#Brooklyn Bridge
plt.subplot(grid[0,0])
plt.plot(date,brookBrd)
ax = plt.gca()
plt.ylim([0, 10000])
ax.set_yticks([0, 2500, 5000, 7500, 10000])
plt.xticks(rotation=90)
locator = AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(AutoDateFormatter(locator) )
plt.xlabel('Time')
plt.ylabel('Bike Usage')
ax.set_title('Brooklyn Bridge')
plt.grid()

#Manhattan Bridge
plt.subplot(grid[0,1])
plt.plot(date,manBrd)
ax = plt.gca()
plt.ylim([0, 10000])
ax.set_yticks([0, 2500, 5000, 7500, 10000])
plt.xticks(rotation=90)
locator = AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(AutoDateFormatter(locator) )
plt.xlabel('Time')
plt.ylabel('Bike Usage')
ax.set_title('Manhattan Bridge')
plt.grid()

#Queensboro Bridge
plt.subplot(grid[1,0])
plt.plot(date,queenBrd)
ax = plt.gca()
plt.ylim([0, 10000])
ax.set_yticks([0, 2500, 5000, 7500, 10000])
plt.xticks(rotation=90)
locator = AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(AutoDateFormatter(locator) )
plt.xlabel('Time')
plt.ylabel('Bike Usage')
ax.set_title('Queensboro Bridge')
plt.grid()

#Williamsburg Bridge
plt.subplot(grid[1,1])
plt.plot(date,williamsBrd)
ax = plt.gca()
plt.ylim([0, 10000])
ax.set_yticks([0, 2500, 5000, 7500, 10000])
plt.xticks(rotation=90)
locator = AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(AutoDateFormatter(locator) )
plt.xlabel('Time')
plt.ylabel('Bike Usage')
ax.set_title('Williamsburg Bridge')
plt.grid()

#Subplot Config
plt.suptitle('New York City Bridges Bike Usage', fontsize  = 'xx-large')
plt.tight_layout()

plt.show()