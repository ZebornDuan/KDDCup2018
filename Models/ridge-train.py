import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

air_quality = pd.read_csv('E:/working/DataMining/beijing_17_18_aq_cleaned.csv')
air_quality_set = set(air_quality['stationId'].value_counts().to_dict().keys())

for station in air_quality_set:
    aq_station = air_quality[air_quality['stationId'] == station]
    aq_station.to_csv('E:/working/DataMining/%s.csv' % station)


from sklearn import linear_model as lm
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

def score(estimator, X, y):
    y_prediction = estimator.predict(X)
    return np.sum(np.abs(y_prediction - y) / (np.abs(y_prediction) + np.abs(y))) / y.shape[0]

for station in air_quality_set:
    aq = pd.read_csv('E:/working/DataMining/%s.csv' % station)
    for p in ['PM2.5', 'PM10', 'O3']:
        array = np.array(aq[p])
        x = np.array([])
        y = np.array([])

        for i in range(array.shape[0] - 168):
            x_ = array[i: i + 120]
            y_ = array[i + 120: i + 168]
            x = np.append(x, x_)
            y = np.append(y, y_)

        x = x.reshape((-1, 120))
        y = y.reshape((-1, 48))
        
        ridge_param={'alpha': [0.4, 0.5, 0.6, 0.7]}
        grid = GridSearchCV(estimator = lm.Ridge(), param_grid=ridge_param, scoring=score, cv=5)

        grid.fit(x, y)
        joblib.dump(grid, 'E:/working/DataMining/%s_%s.pkl' % (station, p))



data = pd.read_csv('E:/working/DataMining/London_aq.csv')
data = data.dropna(thresh=3)
data[['PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'NO2 (ug/m3)']] = \
data[['PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'NO2 (ug/m3)']].fillna(data[['PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'NO2 (ug/m3)']].mean())

data = data.rename(columns = {'PM2.5 (ug/m3)': 'PM2.5', 'PM10 (ug/m3)':'PM10'})

data_set = set(data['station_id'].value_counts().to_dict().keys())

for station in data_set:
    aq_station = data[data['station_id'] == station]
    aq_station.to_csv('E:/working/DataMining/%s.csv' % station)

for station in data_set:
    aq = pd.read_csv('E:/working/DataMining/%s.csv' % station)
    for p in ['PM2.5', 'PM10']:
        array = np.array(aq[p])
        x = np.array([])
        y = np.array([])

        for i in range(array.shape[0] - 168):
            x_ = array[i: i + 120]
            y_ = array[i + 120: i + 168]
            x = np.append(x, x_)
            y = np.append(y, y_)

        x = x.reshape((-1, 120))
        y = y.reshape((-1, 48))
        
        ridge_param={'alpha': [0.4, 0.5, 0.6, 0.7]}
        grid = GridSearchCV(estimator = lm.Ridge(), param_grid=ridge_param, scoring=score, cv=5)

        grid.fit(x, y)
        joblib.dump(grid, 'E:/working/DataMining/%s_%s.pkl' % (station, p))