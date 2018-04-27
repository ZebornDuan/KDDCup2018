import numpy as np
import pandas as pd
import datetime

path = 'E:/working/DataMining/{}'

def load_data():
	aq1 = pd.read_csv(path.format('beijing_17_18_aq.csv'))
	aq2 = pd.read_csv(path.format('beijing_201802_201803_aq.csv'))
	aq = pd.concat([aq1, aq2], ignore_index=True)
	aq['time'] = pd.to_datetime(aq['utc_time'])
	aq.setindex('time', inplace=True)
	aq.drop('utc_time', axis=1, inplace=True)

	stations = set(aq['stationId'])
	aq_stations = {}
	for s in stations:
		aq_s = aq[aq['stationId'] == s].copy()
		aq_s.drop('stationId', axis=1, inplace=True)
		index_c = aq_s.columns.values.tolist()
		rename_c = {index: s + '_' + index for index in index_c}
		aq_s_ = aq_s.rename(index=str, columns=rename_c)
		aq_stations[s] = aq_s_

	aq_merged = pd.concat(list(aq_stations.values()), axis=1)
	aq_merged['order'] = pd.Series(range(aq_merged.shape[0]), index=aq_merged.index)

	return aq, stations, aq_stations, aq_merged


def generate_dataset(merged, batch_size, dx, dy=48, length=1):
	x_dataset = []
	y_dataset = []

	d = dx + dy
	for i in range(0, merged.shape[0] - d, step):
		x = merged.ix[i:i + dx].values
		y = merged.ix[i + dx, i + d].values

		if batch_size != 1:
			x = np.expand_dims(x, axis=0)
			y = np.expand_dims(y, axis=0)
		if True in np.isnan(x) or True in np.isnan(y):
			continue
		else:
			x_dataset.append(x)
			y_dataset.append(y)

		if batch_size == 1:
			return x_dataset, y_dataset
		else:
			batch_x = []
			batch_y = []
			batch_number = len(x_dataset) // d
			for j in range(batch_number):
				xt = x_dataset[j * d:(j + 1) * d]
				yt = y_dataset[j * d:(j + 1) * d]
				batch_x.append(np.concatenate((xt), axis=0))
				batch_y.append(np.concatenate((yt), axis=0))
			return batch_x, batch_y


def lstm_data(time_series, periods=120, f=4):
	ts = np.array(time_series)
	x_data = ts[:(len(ts) - (len(ts) % periods))]
	y_data = ts[f:(len(ts) - (len(ts) % periods) + f)]

	batch_x = x_data.reshape(-1, periods, 1)
	batch_y = y_data.reshape(-1, periods, 1)

	x_test = ts[-(periods + f):][:periods].reshape(-1, periods, 1)
	y_test = ts[-periods:].reshape(-1, periods, 1)

	return batch_x, batch_y, x_test, y_test

def generate_train_samples(x, y, batch_size=32, periods=168, f_horizon=48):
    total_start_points = len(x) - periods - f_horizon
    start_x_index = np.random.choice(range(total_start_points), batch_size, replace = False)
    
    input_batch_indexs = [list(range(i, i + periods)) for i in start_x_index]
    input_sequence = np.take(x, input_batch_indexs, axis = 0)
    
    output_batch_indexs = [list(range(i + f_horizon, i + periods + f_horizon)) for i in start_x_index]
    output_sequence = np.take(y, output_batch_indexs, axis = 0)
    
	return input_sequence, output_sequence