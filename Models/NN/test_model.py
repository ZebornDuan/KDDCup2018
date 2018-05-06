from model import seq2seq

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes

X, y, predict, global_step = seq2seq()

import numpy as np
import pandas as pd

aq = pd.read_csv('/home/duanchx/KDDCup2018/beijing_201802_201803_aq.csv')
aq[['PM2.5', 'PM10', 'O3']] = aq[['PM2.5', 'PM10', 'O3']].fillna(aq[['PM2.5', 'PM10', 'O3']].mean())
fs = np.array(aq[aq['stationId'] == 'fangshan_aq']['PM2.5'])
x_ = np.expand_dims(fs[0:120], axis=0)
y_ = fs[120:168]

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, './save/iteraction1650')
    feed = {X[t]:x_.reshape((-1, 120))[:,t].reshape((-1, 1)) for t in range(120)}
    feed.update({y[t]: np.array([0.0]).reshape((-1, 1)) for t in range(48)})
    p = session.run(predict, feed_dict=feed)
    p = [np.expand_dims(p_, axis=1) for p_ in p]
    p = np.concatenate(p, axis=1).reshape(48)
    print(np.sum(np.abs(p - y_) / (np.abs(p) + np.abs(y_))) / y_.shape[0])
    print(p)
    print(y_)
