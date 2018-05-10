import requests
import pandas as pd
import numpy as np
import datetime
import math

from model import seq2seq
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes

today = datetime.date.today()
yesterday = (today - datetime.timedelta(days=1)).__format__('%Y-%m-%d')
fromday = (today - datetime.timedelta(days=6)).__format__('%Y-%m-%d')

url = 'https://biendata.com/competition/airquality/bj/%s-0/%s-23/2k0d1d8' % (fromday, yesterday)
file_name = '/home/duanchx/KDDCup2018/bj_airquality_%s-0-%s-23.csv' % (fromday, yesterday)

respones= requests.get(url)
with open (file_name,'w') as f:
    f.write(respones.text)
df = pd.read_csv(file_name)

df = df.rename(columns = {'PM25_Concentration': 'PM2.5', 'PM10_Concentration':'PM10' , 'O3_Concentration':'O3'})
# df.head()
s = pd.read_csv('/home/duanchx/KDDCup2018/sample_submission.csv')
df_set = set(df['station_id'].value_counts().to_dict().keys())

X, y, predict, global_step = seq2seq()

for station in df_set:
    aq = df[df['station_id'] == station]
    for p in ['PM2.5', 'PM10', 'O3']:
        array = np.array(aq[p])[-120:]
        x_ = np.expand_dims(array, axis=0)
        save_weight = './save/iteraction_%s_%s_150' % (station[:-3], p)
        # try:
        # 	open(save_weight, 'r')
        # except IOError:
        # 	save_weight = './save/iteraction_%s_%s_150' % (station[:-3], p)
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(session, save_weight)
            feed = {X[t]:x_.reshape((-1, 120))[:,t].reshape((-1, 1)) for t in range(120)}
            feed.update({y[t]: np.array([0.0]).reshape((-1, 1)) for t in range(48)})
            y_predict = session.run(predict, feed_dict=feed)
            print(y_predict)
            y_predict = [np.expand_dims(p_, axis=1) for p_ in y_predict]
            y_predict = np.concatenate(y_predict, axis=1).reshape(48)
            # print(y_predict)
            for i in range(48):
                try:
                	s.loc[s[s['test_id'] == '%s#%d' % (station, i)].index[0], p] = y_predict[i] if y_predict[i] > 0 else abs(y_predict[i])
                	# print(station, i, y_predict[i])
                except:
                    print(station, i)

url = 'https://biendata.com/competition/airquality/ld/%s-0/%s-23/2k0d1d8' % (fromday, yesterday)
file_name = '/home/duanchx/KDDCup2018/ld_airquality_%s-0-%s-23.csv' % (fromday, yesterday)

respones= requests.get(url)
with open (file_name,'w') as f:
    f.write(respones.text)
df = pd.read_csv(file_name)

df = df.rename(columns = {'PM25_Concentration': 'PM2.5', 'PM10_Concentration':'PM10'})
df[['PM2.5', 'PM10']] = df[['PM2.5', 'PM10']].fillna(df[['PM2.5', 'PM10']].mean())
# df.head()
df_set = set(df['station_id'].value_counts().to_dict().keys())
df_set = ['CD1', 'BL0', 'GR4', 'MY7', 'HV1', 'GN3', 'GR9', 'LW2', 'GN0', 'KF1', 'CD9', 'ST5', 'TH4']

for station in df_set:
    aq = df[df['station_id'] == station]
    for p in ['PM2.5', 'PM10']:
        array = np.array(aq[p])[-120:]
        x_ = np.expand_dims(array, axis=0)
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            try:
            	saver.restore(session, './save/iteraction_%s_%s_250' % (station, p))
            except:
            	saver.restore(session, './save/iteraction_%s_%s_150' % (station, p))
            feed = {X[t]:x_.reshape((-1, 120))[:,t].reshape((-1, 1)) for t in range(120)}
            feed.update({y[t]: np.array([0.0]).reshape((-1, 1)) for t in range(48)})
            y_predict = session.run(predict, feed_dict=feed)
            y_predict = [np.expand_dims(p_, axis=1) for p_ in y_predict]
            y_predict = np.concatenate(y_predict, axis=1).reshape(48)
            # print(y_predict)
            for i in range(48):
                try:
                    s.loc[s[s['test_id'] == '%s#%d' % (station, i)].index[0], p] = y_predict[i] if y_predict[i] > 0 else abs(y_predict[i])
                except:
                    print(station, i)


# In[8]:


s.set_index('test_id').to_csv('/home/duanchx/KDDCup2018/sample_submission.csv')


# In[9]:


files={'files': open('/home/duanchx/KDDCup2018/sample_submission.csv','rb')}
 
data = {
    "user_id": "dcx15",   #user_id is your username which can be found on the top-right corner on our website when you logged in.
    "team_token": "c6538eb842d4bc737b95e09bd04f058f09f8209df94111fa2da5a1122034cc50", #your team_token.
    "description": 'submission',  #no more than 40 chars.
    "filename": "sample_submission.csv", #your filename
}
 
url = 'https://biendata.com/competition/kdd_2018_submit/'
 
response = requests.post(url, files=files, data=data)
 
print(response.text)


# In[ ]:

import os

os.remove('/home/duanchx/KDDCup2018/bj_airquality_%s-0-%s-23.csv' % (fromday, yesterday))
os.remove('/home/duanchx/KDDCup2018/ld_airquality_%s-0-%s-23.csv' % (fromday, yesterday))





