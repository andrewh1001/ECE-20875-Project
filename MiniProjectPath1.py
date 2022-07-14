from audioop import avg
from matplotlib import gridspec
import pandas
import matplotlib.pyplot as plt
from datetime import datetime 
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from yellowbrick.regressor import ResidualsPlot

def PolyCoefficients(x, coeffs):
    o = len(coeffs)
    print(f'# This is a polynomial of order {o}.')
    y = 0
    for i in range(o):
        y += coeffs[i]*x**i
    return y

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
total = brookBrd + manBrd + williamsBrd + queenBrd
avgTemp = (highTemp + lowTemp) / 2
independent = dataset_1[['High Temp', 'Low Temp', 'Precipitation']]
independentAvg = [avgTemp, precipt]
independentAvg = np.array(independentAvg).T
#print(np.shape(independentAvg))
'''
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
'''

#r = np.corrcoef(precipt, total)
#print(r)

'''
X_train, X_test, y_train, y_test = train_test_split(independent, total, test_size = 0.1, random_state = 0)
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred_train = regr.predict(X_train)
y_pred_test = regr.predict(X_test)
intercept = regr.intercept_
coefficient = regr.coef_
score = regr.score(X_test, y_test)
MSE_train = mean_squared_error(y_train, y_pred_train)
MSE_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)

print(r2_train)
print('Score: ', score)
print('Accuracy: ' + str(score*100) + '%')

#normalize data
X_train_mean = np.mean(X_train, axis = 0)
X_train_std = np.std(X_train, axis = 0)
y_mean = np.mean(total)
y_std = np.std(total)
X_train_normalize = (X_train - X_train_mean) / X_train_std
y_normalize = (y_train - y_mean) / y_std
regr.fit(X_train_normalize, y_normalize)

X_test_normalize = (X_test - X_train_mean) / X_train_std
y_test_normalize = (y_test - y_mean) / y_std

y_pred_train = regr.predict(X_train_normalize)
y_pred_test = regr.predict(X_test_normalize)
intercept = regr.intercept_
coefficient = regr.coef_
score = regr.score(X_test_normalize, y_test_normalize)
MSE_train = mean_squared_error(y_normalize, y_pred_train)
MSE_test = mean_squared_error(y_test_normalize, y_pred_test)
r2_train = r2_score(y_normalize, y_pred_train)

print(r2_train)
print('Score: ', score)
print('Accuracy: ' + str(score*100) + '%')
'''

X_train, X_test, y_train, y_test = train_test_split(independent, total, test_size = 0.1, random_state = None)

'''
r2 = []
colors = ['gold', 'yellow', 'red', 'blue', 'purple', 'pink', 'black', 'orange']
degree = [1,2,3,4,5,6,7,8]
print(np.shape(X_train))
print(np.shape(y_train))
'''

#regr = make_pipeline(PolynomialFeatures(8), StandardScaler(), Ridge())
scaled_X_train = StandardScaler().fit_transform(X_train)
scaled_X_test = StandardScaler().fit_transform(X_test)
regr = Ridge(alpha = 0.5)

visualizer = ResidualsPlot(regr, hist=False, qqplot=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

regr.fit(scaled_X_train, y_train)
y_pred_test = regr.predict(scaled_X_test)
print(regr.score(scaled_X_test, y_test))
print(r2_score(y_test, y_pred_test))
print(regr.coef_)
#print(np.shape(X_train))
#print(np.shape(y_pred_train))

'''
for index, num in enumerate(degree):
    regr = make_pipeline(PolynomialFeatures(num), StandardScaler(), Ridge())
    regr.fit(X_train, y_train)
    y_pred_train = regr.predict(X_train)
    y_pred_test = regr.predict(X_test)
    MSE_train = mean_squared_error(y_train, y_pred_train)
    MSE_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2.append(r2_train)

for i in range(len(r2)):
  print('Polynomial Degree ' + str(i+1) + ' r2: ' + str(r2[i]))
'''

'''
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')
axes = [ax1, ax2, ax3]
for ax in axes:
    ax.plot(avgTemp, precipt, total, marker = 'o', linestyle='none')
    ax.plot(X_train[:,0], X_train[:,1], y_pred_train)
'''

plt.show()

