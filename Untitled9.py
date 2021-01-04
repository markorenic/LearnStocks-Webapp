
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from datetime import datetime


#Get the stock quote 
#df = web.DataReader('APPL', data_source='yahoo', start='2019-01-01', end='2020-12-29') 
start = datetime(2010, 1, 1)
end = datetime(2015, 5, 9)
# yahoo will adjust for dividends by default
df = web.DataReader("AAPL", "yahoo", start, end)

# #Visualize closing price
# plt.figure(figsize=(16,8))
# plt.title('Closing Price history')
# plt.plot(df['Close'])
# plt.xlabel('Data',fontsize = 18)
# plt.ylabel('Close Price USD ($)', fontsize = 18)
# plt.show()


#Create moving average
SMA30 = pd.DataFrame()
SMA30['Adj Close Price'] = df['Adj Close'].rolling(window = 30).mean()
#Create moving average
SMA100 = pd.DataFrame()
SMA100['Adj Close Price'] = df['Adj Close'].rolling(window = 100).mean()
SMA100
#Create Short-term EMA
SEMA12 = df.Close.ewm(span=7,adjust= False).mean()
#Create Long-term EMA
LEMA26 = df.Close.ewm(span=31,adjust= False).mean()

#calculate MACD
MACD = SEMA12 - LEMA26
signal = MACD.ewm(span=9,adjust=False).mean()


# #Visualize closing price
# plt.figure(figsize=(16,8))
# plt.title('Closing Price history')
# plt.plot(df['Adj Close'], label = 'APPL')
# plt.plot(SMA30['Adj Close Price'], label = 'SMA30')
# plt.plot(SMA100['Adj Close Price'], label = 'SMA100')
# plt.plot(df.index,MACD,label='MACD Line', color ='Green')
# plt.plot(df.index,signal,label='Signal Line', color ='Purple')
# plt.xlabel('Data',fontsize = 18)
# plt.ylabel('Close Price USD ($)', fontsize = 18)
# plt.legend(loc = 'upper left')
# plt.show()

#Create new df with all the data
data = pd.DataFrame()
data['AAPL'] = df['Adj Close']
data['SMA30'] = SMA30['Adj Close Price']
data['SMA100'] = SMA100['Adj Close Price']
data['MACD'] = MACD
data['Signal Line'] = signal

#Function to send buy/sell signal
def buy_sell(data):
  sigPriceBuy = []
  sigPriceSell = []
  flag = -1
  for i in range(len(data)):
    if data['MACD'][i] > data['Signal Line'][i]: #data['SMA30'][i]>data['SMA100'][i] or 
      if flag != 1:
        sigPriceBuy.append(data['AAPL'][i])
        sigPriceSell.append(np.nan)
        flag = 1
      else:
        sigPriceBuy.append(np.nan)
        sigPriceSell.append(np.nan)
    elif data['MACD'][i] < data['Signal Line'][i]: #data['SMA30'][i] < data['SMA100'][i] or 
      if flag != 0:
        sigPriceBuy.append(np.nan)
        sigPriceSell.append(data['AAPL'][i])
        flag = 0
      else:
        sigPriceBuy.append(np.nan)
        sigPriceSell.append(np.nan)
    else:
        sigPriceBuy.append(np.nan)
        sigPriceSell.append(np.nan)

  return (sigPriceBuy,sigPriceSell)

#Store by and sell
buy_sell = buy_sell(data)
data['Buy_Signal'] = buy_sell[0]
data['Sell_Signal'] = buy_sell[1]

#Visualize closing price
plt.figure(figsize=(16,8))
plt.title('Closing Price history')
plt.plot(df['Adj Close'], label = 'APPL' , alpha = 0.5)
plt.plot(SMA30['Adj Close Price'], label = 'SMA30' , alpha = 0.5)
plt.plot(SMA100['Adj Close Price'], label = 'SMA100' , alpha = 0.5)
plt.scatter(data.index,data['Buy_Signal'],label = 'Buy', marker = '^',color = 'green')
plt.scatter(data.index,data['Sell_Signal'],label = 'Sell', marker = 'v',color = 'red')
plt.xlabel('Data',fontsize = 18)
plt.plot(df.index,MACD,label='MACD Line', color ='Green')
plt.plot(df.index,signal,label='Signal Line', color ='Purple')

plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.legend(loc = 'upper left')
plt.show()



