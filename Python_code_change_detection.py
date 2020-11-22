# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pylab as plt
import ruptures as rpt
import pandas as pd
import urllib.request
import json
import os

#%%

# User parameters
Mkt_Ticker = 'AAPL' # Ticker Name
## time_interval = 30min # timeinterval for ticker Line 35;46

nchng = 8 # number of changes for change detection
frequency = 'INTRADAY' # DAILY vs INTRADAY
timeinterval = '60min' ## For Intraday 30 min time interval data

dailyoutputsize = 'full' #compact or full

ncp = 'KNOWN' # Number of Change points KNOWN UNKNOWN
sigma = .5 # sigma level (standard devition) in case of UNKNOWN change points


## Limitations : May not detect parbolic moves
## Limitations : No Sell Signal if continually rising
## Limitations : May cause lower returns if too many swings in rising price movements
## Falling Markets : May cause lower loss in falling markets

#%%

def import_web(ticker):
    """
    :param identifier: List, Takes the company name
    :return:displays companies records per minute
    """
#    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol='+ticker +'&interval=5min&apikey=LJLI2YHYQLUYR0L5&outputsize=full&datatype=json'
    if frequency == 'INTRADAY':    
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol='+ticker +'&interval='+timeinterval+'&apikey=LJLI2YHYQLUYR0L5&outputsize=full&datatype=json'
    if frequency == 'DAILY':  
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+ticker +'&outputsize='+dailyoutputsize+'&apikey=LJLI2YHYQLUYR0L5l&datatype=json'
    fp = urllib.request.urlopen(url)
    mybytes = fp.read()
    mystr = mybytes.decode("utf8")
    fp.close()
    return mystr


def get_value(ticker):
    js = import_web(ticker)
    parsed_data = json.loads(js) # loads the json and converts the json string into dictionary
#    ps = parsed_data['Time Series (5min)']
    if frequency == 'INTRADAY':
        ps = parsed_data['Time Series ('+timeinterval+')']
    if frequency == 'DAILY':
        ps = parsed_data['Time Series (Daily)']
    return ps
#    partitionSave(ps,ticker)


#%%
    
# Data Transformations

mydata = get_value(Mkt_Ticker)
df = pd.DataFrame.from_dict(mydata, orient='columns')
df_transposed = df.T
df_transposed = df_transposed.astype(float).sort_index(axis=0)

#%%

# Creating the time series signal matrix

signal = np.array(df_transposed['4. close'])

data = signal

data = pd.DataFrame(data)
data['DateTime'] = df_transposed.index
data.reset_index(level=0, inplace=True)
data = data.drop('index', axis =1)

data.columns = ['4. close', 'DateTime']

n=len(data)

#%%
data = data.sort_index(axis=0, level=None, ascending=False)
data.index = range(n)

#%%

# Applying change point detection algorithm

model = "l2"  # "l1", "rbf", "linear", "normal", "ar"
algo = rpt.Binseg(model=model).fit(signal)
if ncp == 'KNOWN':
    my_bkps = algo.predict(n_bkps=nchng)
if ncp == 'UNKNOWN':
    my_bkps = algo.predict(epsilon=3*n*sigma**2)
    
#%%

# Show results, Plotting
rpt.show.display(signal, my_bkps, figsize=(10, 6))

plt.show()

#%%

# Notification for Change-Point
notification_ind = pd.Series(my_bkps)-1
data['Status'] = 0 # 0 : Previous State
data.loc[notification_ind,'Status'] = 100 # 100 : New State

for i in np.arange(0, len(notification_ind)-1):    
    a = np.mean(data.loc[notification_ind[i]-4:notification_ind[i],'4. close'])
    b = np.mean(data.loc[notification_ind[i]+1:notification_ind[i]+4,'4. close'])
    #print (a,b)
    if a > b:
        data.loc[notification_ind[i],'Status'] = -100
    elif a < b:
        data.loc[notification_ind[i],'Status'] = 100

#%%
