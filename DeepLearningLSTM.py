#!/usr/bin/env python
# coding: utf-8

"""
Created on Fri Jun 7 16:39:34 2020

@author: sovik

Algorithm Name :: Anomaly Detection (ad) Using Deep Learning LSTM

"""

#%%

# libraries

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn
import matplotlib.dates as md
from matplotlib import pyplot as plt
#from mpl_toolkits.axes_grid1 import host_subplot
#import mpl_toolkits.axisartist as AA
from sklearn import preprocessing

# specific libraries for RNN
# keras is a high layer build on Tensorflow layer to stay in high level/easy implementation
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
#import lstm, time #helper libraries
import time
from keras.models import model_from_json

### PYTHON 3.6 SERIES NEEDED; TENSORFLOW DOESNOT WORK WITH PYTHON 3.7 <<<-----------

#%%

## Algorithm parameters <<-- USER INPUT IN SELF DISCOVERY PORTAL

outliers_fraction = 0.01 # An estimation of anomly population of the dataset (necessary for several algorithm)

## Deep Learning parameters

# Train/Test Data Split Parameters
prediction_time = 1 
testdatasize = 22500 # 22000
unroll_length = 50

# Model Training Parameters
train_batch_size=3028 # Increases in batchsize inceases data passed for each epoch of training
train_nb_epoch=3 # Number of times the model is trained
train_validation_split=0.1

# Model Activation Function
activation_DL = 'linear'

# Model compilation, Loss function and optimizer
loss_DL='mse'
optimizer_DL='rmsprop'

#%%

## Read Data  <<-- FROM SELF DISCOVERY PORTAL

tsdata = pd.read_csv("machine_temperature_system_failure.csv")

#%%

datapred = tsdata[['Date', 'value']]
df = tsdata[['Date', 'value']]

#%%

## Data processing and feature engineering

# change the type of timestamp column for plotting
df['timestamp'] = pd.to_datetime(df['Date'])

# the hours and if it's night or day (7:00-22:00)
df['hours'] = df['timestamp'].dt.hour
#df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)

# the day of the week (Monday=0, Sunday=6) and if it's a week end day or week day.
df['DayOfTheWeek'] = df['timestamp'].dt.dayofweek
df['WeekDay'] = (df['DayOfTheWeek'] < 5).astype(int)
# An estimation of anomly population of the dataset (necessary for several algorithm)

# Take useful feature and standardize them
#data_n = df[['value', 'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]
data_n = df[['value', 'hours', 'DayOfTheWeek', 'WeekDay']]
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data_n)
data_n = pd.DataFrame(np_scaled)

testdatacut = testdatasize + unroll_length  + 1

#train data <<<--- Need to figure out this
x_train = data_n[0:-prediction_time-testdatacut].as_matrix()
y_train = data_n[prediction_time:-testdatacut  ][0].as_matrix()

# test data <<<--- Need to figure out this
x_test = data_n[0-testdatacut:-prediction_time].as_matrix()
y_test = data_n[prediction_time-testdatacut:  ][0].as_matrix()

#unroll: create sequence of 50 previous data points for each data points
def unroll(data,sequence_length=24):
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    return np.asarray(result)

# adapt the datasets for the sequence data shape
x_train = unroll(x_train,unroll_length)
x_test  = unroll(x_test,unroll_length)
y_train = y_train[-x_train.shape[0]:]
y_test  = y_test[-x_test.shape[0]:]


#%%

# Build the Deep Learning LSTM model
model = Sequential()

model.add(LSTM(
    input_dim=x_train.shape[-1],
    output_dim=50,
    return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    100,
    return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(
    output_dim=1))
model.add(Activation(activation_DL))

start = time.time()
model.compile(loss=loss_DL, optimizer=optimizer_DL)
#print('compilation time : {}'.format(time.time() - start))


#%%


# Train the model

model.fit(
    x_train,
    y_train,
    batch_size=train_batch_size,
    nb_epoch=train_nb_epoch,
    validation_split=train_validation_split)



#%%

# save the model because the training is long

"""
# serialize model to JSON
model_json = model.to_json()
with open("model2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model2.h5")
print("Saved model to disk")
"""


#%%

# Load saved model

"""
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
"""

#%%

# create the list of difference between prediction and test data
diff=[]
ratio=[]
p = model.predict(x_test)
# predictions = lstm.predict_sequences_multiple(loaded_model, x_test, 50, 50)
for u in range(len(y_test)):
    pr = p[u][0]
    ratio.append((y_test[u]/pr)-1)
    diff.append(abs(y_test[u]- pr))



#%%

# select the most distant prediction/reality data points as anomalies
diff = pd.Series(diff)
number_of_outliers = int(outliers_fraction*len(diff))
threshold = diff.nlargest(number_of_outliers).min()
# data with anomaly label (test data part)
test = (diff >= threshold).astype(int)
# the training data part where we didn't predict anything (overfitting possible): no anomaly
complement = pd.Series(0, index=np.arange(len(data_n)-testdatasize))
# # add the data to the main
df['anomalyDL_LSTM'] = complement.append(test, ignore_index='True')
#print(df['anomalyDL_LSTM'].value_counts())


#%%

# Defining Anomaly labels for Predicted Anomaly

datapred['PredictedAnomaly'] = 'FALSE' # 0 : Previous State
datapred.loc[df['anomalyDL_LSTM'] == 1,'PredictedAnomaly'] = 'TRUE' # 100 : New State


#%%

## Visualizations
#
## plot the prediction and the reality (for the test data)
#fig, axs = plt.subplots()
#axs.plot(p,color='red', label='prediction')
#axs.plot(y_test,color='blue', label='y_test')
#plt.legend(loc='upper left')
#plt.show()
#
## visualisation of anomaly throughout time (viz 1)
##fig, ax = plt.subplots()
##
##a = df.loc[df['anomalyDL_LSTM'] == 1, ['time_epoch', 'value']] #anomaly
##
##ax.plot(df['time_epoch'], df['value'], color='blue')
##ax.scatter(a['time_epoch'],a['value'], color='red')
##plt.show()
#
#
#a = df.loc[df['anomalyDL_LSTM'] == 1, ['timestamp', 'value']] #anomaly
#
#plt.figure()
#plt.plot(df['timestamp'], df['value'], 'b-')
#plt.plot(a['timestamp'],a['value'], 'ro')
#plt.show()


#%%


## visualisation of anomaly with temperature repartition (viz 2)
#a = df.loc[df['anomalyDL_LSTM'] == 0, 'value']
#b = df.loc[df['anomalyDL_LSTM'] == 1, 'value']

#fig, axs = plt.subplots()
#axs.hist([a,b], bins=32, stacked=True, color=['blue', 'red'])
#plt.legend()
#plt.show()




