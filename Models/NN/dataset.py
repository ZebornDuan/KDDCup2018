import numpy as np
import pandas as pd

path = 'E:/working/DataMining/{}'

aq = pd.read_csv(path.format('beijing_17_18_aq_cleaned.csv'))
array = np.array(aq['PM2.5'])

batch_size = 20

def generate():
	x = np.array([])
	y = np.array([])
	n = 0
	for i in range(array.shape[0] - 168):
		x_ = array[i: i + 120]
		y_ = array[i + 120: i + 168]
		x = np.append(x, x_)
		y = np.append(y, y_)
		if n % batch_size == 0:
			yield x, y
			x = np.array([])
			y = np.array([])
			n = 0