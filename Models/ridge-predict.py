
# coding: utf-8

# In[1]:


import requests
import pandas as pd
import numpy as np
import datetime

today = datetime.date.today()

yesterday = (today - datetime.timedelta(days=1)).__format__('%Y-%m-%d')

fromday = (today - datetime.timedelta(days=6)).__format__('%Y-%m-%d')

url = 'https://biendata.com/competition/airquality/bj/%s-0/%s-23/2k0d1d8' % (fromday, yesterday)
file_name = 'E:/bj_airquality_%s-0-%s-23.csv' % (fromday, yesterday)

respones= requests.get(url)
with open (file_name,'w') as f:
    f.write(respones.text)
df = pd.read_csv(file_name)

df = df.rename(columns = {'PM25_Concentration': 'PM2.5', 'PM10_Concentration':'PM10' , 'O3_Concentration':'O3'})
# df.head()


# In[2]:


df[['PM2.5', 'PM10', 'O3']] = df[['PM2.5', 'PM10', 'O3']].fillna(df[['PM2.5', 'PM10', 'O3']].mean())
# df.head()


# In[3]:


from sklearn.externals import joblib
s = pd.read_csv('E:/sample_submission.csv')

df_set = set(df['station_id'].value_counts().to_dict().keys())
df_set

# submit_index = {i:i for i in df_set}
# submit_index['aotizhongxin_aq'] = 'aotizhongx_aq'
# submit_index['xizhimenbei_aq'] = 'xizhimenbe_aq'
# submit_index['wanshouxigong_aq'] = 'wanshouxig_aq'
# submit_index['miyunshuiku_aq'] = 'miyun_aq'
# submit_index['nongzhanguan_aq'] = 'nongzhangu_aq'
# submit_index['yongdingmennei_aq'] = 'yongdingme_aq'
# submit_index['fengtaihuayuan_aq'] = 'fengtaihua_aq'


# In[4]:


def score(estimator, X, y):
    y_prediction = estimator.predict(X)
    return np.sum(np.abs(y_prediction - y) / (np.abs(y_prediction) + np.abs(y))) / y.shape[0]

for station in df_set:
    aq = df[df['station_id'] == station]
    for p in ['PM2.5', 'PM10', 'O3']:
        array = np.array(aq[p])[-120:]
        grid = joblib.load('E:/working/DataMining/%s_%s.pkl' % ('dongsi_aq', 'PM2.5'))
        y = grid.predict(np.expand_dims(array, axis=0))
        for i in range(48):
            try:
                s.loc[s[s['test_id'] == '%s#%d' % (station, i)].index[0], p] = y[0][i] if y[0][i] > 0 else 0
            except:
                print(station, i)
# s.head()


# In[5]:


url = 'https://biendata.com/competition/airquality/ld/%s-0/%s-23/2k0d1d8' % (fromday, yesterday)
file_name = 'E:/ld_airquality_%s-0-%s-23.csv' % (fromday, yesterday)

respones= requests.get(url)
with open (file_name,'w') as f:
    f.write(respones.text)
df = pd.read_csv(file_name)

df = df.rename(columns = {'PM25_Concentration': 'PM2.5', 'PM10_Concentration':'PM10'})
df[['PM2.5', 'PM10']] = df[['PM2.5', 'PM10']].fillna(df[['PM2.5', 'PM10']].mean())
df.head()


# In[6]:


df_set = set(df['station_id'].value_counts().to_dict().keys())
df_set = ['CD1', 'BL0', 'GR4', 'MY7', 'HV1', 'GN3', 'GR9', 'LW2', 'GN0', 'KF1', 'CD9', 'ST5', 'TH4']


# In[7]:


for station in df_set:
    aq = df[df['station_id'] == station]
    for p in ['PM2.5', 'PM10']:
        array = np.array(aq[p])[-120:]
        grid = joblib.load('E:/working/DataMining/%s_%s.pkl' % ('dongsi_aq', 'PM2.5'))
        y = grid.predict(np.expand_dims(array, axis=0))
        for i in range(48):
            try:
                s.loc[s[s['test_id'] == '%s#%d' % (station, i)].index[0], p] = y[0][i] if y[0][i] > 0 else 0
            except:
                print(station, i)


# In[8]:


s.set_index('test_id').to_csv('E:/sample_submission.csv')


# In[9]:


files={'files': open('E:/sample_submission.csv','rb')}
 
data = {
    "user_id": "dcx15",   #user_id is your username which can be found on the top-right corner on our website when you logged in.
    "team_token": "", #your team_token.
    "description": 'submission',  #no more than 40 chars.
    "filename": "sample_submission.csv", #your filename
}
 
url = 'https://biendata.com/competition/kdd_2018_submit/'
 
response = requests.post(url, files=files, data=data)
 
print(response.text)


# In[ ]:

import os

os.remove('E:/bj_airquality_%s-0-%s-23.csv' % (fromday, yesterday))
os.remove('E:/ld_airquality_%s-0-%s-23.csv' % (fromday, yesterday))



