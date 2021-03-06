import numpy as np
import pandas as pd

path = '/home/duanchx/KDDCup2018/{}'

batch_size = 30

normalization = {'PM2.5': 1000.0, 'PM10': 3000.0, 'O3': 300}

def generate(where, which):
	aq = pd.read_csv(path.format('%s.csv' % where))
	array = np.array(aq[which])
	x = np.array([])
	y = np.array([])
	n = 0
	for i in range(array.shape[0] - 168):
		x_ = array[i: i + 120]
		y_ = array[i + 120: i + 168]
		x = np.append(x, x_)
		y = np.append(y, y_)
		n += 1
		if n % batch_size == 0:
			yield x / normalization[which], y / normalization[which]
			x = np.array([])
			y = np.array([])
			n = 0