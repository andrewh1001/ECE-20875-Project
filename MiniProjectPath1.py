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
brookBrd = dataset_1['Brooklyn Bridge']

plt.figure(1)
fig = plt.plot(date,highTemp)
ax = plt.gca()
plt.xticks(rotation=90)
locator = AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(AutoDateFormatter(locator) )
plt.xlabel('Time')
plt.ylabel('Highest Temperature (Celcius)')
plt.grid()

plt.figure(2)
fig = plt.plot(date,lowTemp)
ax = plt.gca()
plt.xticks(rotation=90)
locator = AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(AutoDateFormatter(locator) )
plt.xlabel('Time')
plt.ylabel('Lowest Temperature (Celcius)')
plt.grid()

plt.figure(3)
fig = plt.plot(date,brookBrd)
ax = plt.gca()
plt.xticks(rotation=90)
locator = AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(AutoDateFormatter(locator) )
plt.xlabel('Time')
plt.ylabel('Bike Usage')
plt.grid()

plt.show()