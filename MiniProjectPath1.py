import pandas
import matplotlib.pyplot as plt
from datetime import datetime 
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from yellowbrick.regressor import ResidualsPlot
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LogisticRegression

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
day = dataset_1['Day']
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

#Question 1
print(f"Williamsburg Bridge Total Traffic: {sum(williamsBrd)} Standard Deviation: {np.std(williamsBrd)} Mean: {np.mean(williamsBrd)}")
print(f"Queensboro Bridge Total Traffic: {sum(queenBrd)} Standard Deviation: {np.std(queenBrd)} Mean: {np.mean(queenBrd)}")
print(f"Manhattan Bridge Total Traffic: {sum(manBrd)} Standard Deviation: {np.std(manBrd)} Mean: {np.mean(manBrd)}")
print(f"Brooklyn Bridge Total Traffic: {sum(brookBrd)} Standard Deviation: {np.std(brookBrd)} Mean: {np.mean(brookBrd)}")

bridges = ['Williamsburg', 'Queensboro', 'Manhattan', 'Brooklyn']
bridges_mean = [np.mean(williamsBrd), np.mean(queenBrd), np.mean(manBrd), np.mean(brookBrd)]
bridges_std = [np.std(williamsBrd), np.std(queenBrd), np.std(manBrd), np.std(brookBrd)]
plt.figure(3)
x_axis = np.arange(len(bridges))
plt.bar(x_axis, bridges_mean, width = 0.25, label = 'Mean')
plt.bar(x_axis + 0.25, bridges_std, width = 0.25, label = "STD")
plt.xticks(x_axis, bridges)
plt.xlabel('Bridges')
plt.ylabel('Number of Cyclists')
plt.title("Mean and STD Traffic for New York Bridges")
plt.legend()
plt.grid(False)
plt.show()
#Question 2
X_train, X_test, y_train, y_test = train_test_split(independent, total, test_size = 0.25, random_state = 80)

#regr = make_pipeline(PolynomialFeatures(8), StandardScaler(), Ridge())
X_train_scaler = StandardScaler().fit(X_train)
X_test_scaler = StandardScaler().fit(X_test)
scaled_X_train = X_train_scaler.transform(X_train)
scaled_X_test = X_test_scaler.transform(X_test)

lmbda = np.logspace(-1, 3, num = 51)
ridge_cv = RidgeCV(alphas = lmbda, scoring = 'neg_mean_squared_error', fit_intercept= True)
ridge_cv.fit(scaled_X_train, y_train)
ridge_alpha = ridge_cv.alpha_
print("alpha: " + str(ridge_alpha))

regr = Ridge(alpha = ridge_alpha)

regr.fit(scaled_X_train, y_train)
y_pred_test = regr.predict(scaled_X_test)
print("r2 score: " + str(r2_score(y_test, y_pred_test)))
print("MSE: " + str(mean_squared_error(y_test, y_pred_test)))
#print(regr.coef_)

visualizer = ResidualsPlot(regr, hist=False, qqplot = True)
visualizer.fit(scaled_X_train, y_train)
visualizer.score(scaled_X_test, y_test)
visualizer.show()

#Question 3
mon_index = day.index[day == 'Monday'].tolist()
tue_index = day.index[day == 'Tuesday'].tolist()
wed_index = day.index[day == 'Wednesday'].tolist()
thur_index = day.index[day == 'Thursday'].tolist()
fri_index = day.index[day == 'Friday'].tolist()
sat_index = day.index[day == 'Saturday'].tolist()
sun_index = day.index[day == 'Sunday'].tolist()

mon_traffic = total[mon_index]
tue_traffic = total[tue_index]
wed_traffic = total[wed_index]
thur_traffic = total[thur_index]
fri_traffic = total[fri_index]
sat_traffic = total[sat_index]
sun_traffic = total[sun_index]

'''
print(np.mean(mon_traffic))
print(np.mean(tue_traffic))
print(np.mean(wed_traffic))
print(np.mean(thur_traffic))
print(np.mean(fri_traffic))
print(np.mean(sat_traffic))
print(np.mean(sun_traffic))
'''

total = total.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(total, day, test_size = 0.25, random_state = 80)
logReg = LogisticRegression(penalty = 'none')
logReg.fit(X_train.reshape(-1, 1), y_train)
print(logReg.score(X_test.reshape(-1, 1), y_test))