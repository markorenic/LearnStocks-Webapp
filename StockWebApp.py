from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import math
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pandas_datareader as web
from datetime import date
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


st.set_option('deprecation.showPyplotGlobalUse', False)


st.header("Stockagram")
st.write("""Scroll down to visualize stock data and indicators as well as test out making predictions using your custome neural-net!""")
option = st.selectbox('What would you like to learn about?',
                      ("About Us", "Moving Average Crossover", 'Moving Average Convergence Divergence (MACD)', "Future price prediction"))


# def get_input():
#     start_date = st.sidebar.text_input("Start Date","2018-01-01")
#     end_date = st.sidebar.text_input("End Date","2020-01-01")
#     stock = st.sidebar.text_input("Stock","TSLA")
#     return start_date,end_date,stock

def get_data(symbol, startt, endt):
    start = startt  # The date - 29 Dec 2017
    end = endt  # The date - 29 Dec 2017
    format_str = '%Y-%m-%d'  # The format
    starttime_obj = datetime.strptime(start, format_str)
    endtime_obj = datetime.strptime(end, format_str)
    df = web.DataReader(symbol, "yahoo", starttime_obj, endtime_obj)
    return df


def get_SMA(len, df):
    SMA = df['Adj Close'].rolling(window=len).mean()
    return SMA


def get_EMA(len, df):
    EMA = df.Close.ewm(span=len, adjust=False).mean()
    return EMA


def get_MACD(sema, lema, df):
    SEMA = get_EMA(sema, df)
    LEMA = get_EMA(lema, df)
    MACD = SEMA - LEMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    return MACD, signal, SEMA, LEMA

# Function to send buy/sell signal


def buy_sell_MACD(data):
    sigPriceBuy = []
    sigPriceSell = []
    flag = -1
    for i in range(len(data)):
        if data['MACD'][i] > data['Signal Line'][i]:  # data['SMA'][i]>data['LMA'][i] or
            if flag != 1:
                sigPriceBuy.append(data[stock][i])
                sigPriceSell.append(np.nan)
                flag = 1
            else:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)
        elif data['MACD'][i] < data['Signal Line'][i]:  # data['SMA'][i] < data['LMA'][i] or
            if flag != 0:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(data[stock][i])
                flag = 0
            else:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)
        else:
            sigPriceBuy.append(np.nan)
            sigPriceSell.append(np.nan)

    return (sigPriceBuy, sigPriceSell)


def buy_sell_SMA(data):
    sigPriceBuy = []
    sigPriceSell = []
    flag = -1
    for i in range(len(data)):
        if data['SMA'][i] > data['LMA'][i]:
            if flag != 1:
                sigPriceBuy.append(data[stock][i])
                sigPriceSell.append(np.nan)
                flag = 1
            else:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)
        elif data['SMA'][i] < data['LMA'][i]:
            if flag != 0:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(data[stock][i])
                flag = 0
            else:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)
        else:
            sigPriceBuy.append(np.nan)
            sigPriceSell.append(np.nan)

    return (sigPriceBuy, sigPriceSell)

# if option != "About Us":
    #st.sidebar.header('Choose Data to test:')
    #start_date,end_date,stock = st.sidebar.text_input("Start Date","2018-01-01"),st.sidebar.text_input("End Date","2020-01-01"),st.sidebar.text_input("Stock","TSLA")
    #df = get_data(stock,start_date,end_date)
    #data = pd.DataFrame()


# pages:
if option == "About Us":
    st.write("Please choose")

# SMA___________________________________
elif option == "Moving Average Crossover":
    # Sidebar
    st.sidebar.header('Choose Data to test:')
    start_date, end_date, stock = st.sidebar.text_input(
        "Start Date", "2018-01-01"), st.sidebar.text_input("End Date", "2020-01-01"), st.sidebar.text_input("Stock", "TSLA")
    df = get_data(stock, start_date, end_date)
    data = pd.DataFrame()
    format_str = '%Y-%m-%d'  # The format
    starttime_obj = datetime.strptime(start_date, format_str)
    endtime_obj = datetime.strptime(end_date, format_str)
    delta = endtime_obj - starttime_obj
    delta = delta.days
    shortSMA = int(st.sidebar.slider(
        'Short-term moving average (Default 30)', 1, 100, 30, 1))
    LongSMA = int(st.sidebar.slider(
        'Long-term moving average (Default 100)', 1, delta, 100, 1))

    # Get data
    data[stock] = df['Adj Close']
    data['SMA'] = get_SMA(shortSMA, df)
    data['LMA'] = get_SMA(LongSMA, df)
    buy_sell_SMA = buy_sell_SMA(data)
    data['SMA_Buy_Signal'] = buy_sell_SMA[0]
    data['SMA_Sell_Signal'] = buy_sell_SMA[1]

    # Page
    st.title("Moving Average Crossover")
    st.write("Investors use moving averages for a plethora of reasons such as a primary analytic tool or as confidence builder to back up their investment decisions. However, no matter how useful moving averages are given the vital data they convey, they have one great limitation – they are a lagging indicator. By the time an 100-day MA curves upward or downward and conveys a trend, the market has already exhausted a part of that move and may even be nearing its end. And although exponential averages speed up signals, all MAs are lagging behind. This is why traders do not base their trading decisions solely on moving averages and generally wait for the strongest possible signals they generate – crossovers. A crossover is when a short-term moving average crosses a long-term moving average, which suggests a shift in the momentum of the stock.")
    st.write("On the sidebar you can change the dataset and play around with the short and longer term averages so see what kind of indicators it will create.\n Note that the green arrow represents a buying signal and red represents a selling singal suggesting a crossover has happened.")
    st.write("Remeber: The goal is to buy low and sell high!")
    
    # Plot
    st.header(stock + " SMA Indicator Signals")
    plt.figure(figsize=(20, 10))
    plt.title('Closing Price history')
    plt.plot(df['Adj Close'], label=stock, alpha=0.5)
    plt.plot(data['SMA'], label='SMA'+str(shortSMA), alpha=0.5)
    plt.plot(data['LMA'], label='LMA'+str(LongSMA), alpha=0.5)
    plt.scatter(data.index, data['SMA_Buy_Signal'],
                label='Buy', marker='^', color='green')
    plt.scatter(data.index, data['SMA_Sell_Signal'],
                label='Sell', marker='v', color='red')
    plt.xlabel('Data', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.legend(loc='upper left')
    st.pyplot()

# MACD__________________________________
elif option == "Moving Average Convergence Divergence (MACD)":

    with st.beta_container():
      st.title("Moving Average Convergence Divergence (MACD)")
      st.write("Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows the relationship between two exponential moving averages of a security’s price. The MACD is calculated by subtracting the 26-period exponential moving average (EMA) from the 12-period EMA. The result of that calculation is the MACD line.\n Moving average convergence divergence (MACD) is calculated by subtracting the 26-period exponential moving average (EMA) from the 12-period EMA. MACD triggers technical signals when it crosses above (to buy) or below (to sell) its signal line. The speed of crossovers is also taken as a signal of a market is overbought or oversold. MACD helps investors understand whether the bullish or bearish movement in the price is strengthening or weakening. Similar to the moving average line used in the Moving average Crossover, the exponential moving average is calculated using the historical data of the security.")
      st.write("Learn more about EMA and MACD: https://www.investopedia.com/terms/e/ema.asp \n https://www.investopedia.com/terms/m/macd.asp")
      st.write("On the sidebar you can change how far back you want the algorithm to look and see what kind of results it yealds!")
      st.write("Remeber: The goal is to buy low and sell high!")

    st.sidebar.header('Choose Data to test:')
    start_date, end_date, stock = st.sidebar.text_input(
        "Start Date", "2018-01-01"), st.sidebar.text_input("End Date", "2020-01-01"), st.sidebar.text_input("Stock", "TSLA")
    df = get_data(stock, start_date, end_date)
    data = pd.DataFrame()

    format_str = '%Y-%m-%d'  # The format
    starttime_obj = datetime.strptime(start_date, format_str)
    endtime_obj = datetime.strptime(end_date, format_str)
    delta = endtime_obj - starttime_obj
    delta = delta.days
    shortEMA = int(st.sidebar.slider(
        'Short-term exponential moving average (Default 12)', 1, 100, 12, 1))
    LongEMA = int(st.sidebar.slider(
        'Long-term exponential moving average (Default 26)', 1, delta, 26, 1))

    # Store by and sell
    data[stock] = df['Adj Close']
    data['MACD'], data['Signal Line'], data['Short-term EMA'], data['Long-term EMA'] = get_MACD(
        shortEMA, LongEMA, df)
    buy_sell_MACD = buy_sell_MACD(data)
    data['MACD_Buy_Signal'] = buy_sell_MACD[0]
    data['MACD_Sell_Signal'] = buy_sell_MACD[1]
    st.header(stock + " MACD Indicator Signals")
    plt.figure(figsize=(16, 8))
    plt.title(stock + " MACD Indicator Signals")
    plt.plot(df['Adj Close'], label=stock, alpha=0.5)
    plt.plot(data['Short-term EMA'], label="Short-term EMA", alpha=0.5)
    plt.plot(data['Long-term EMA'], label="Long-term EMA", alpha=0.5)
    plt.scatter(data.index, data['MACD_Buy_Signal'],
                label='Buy', marker='^', color='green')
    plt.scatter(data.index, data['MACD_Sell_Signal'],
                label='Sell', marker='v', color='red')
    plt.xlabel('Data', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.legend(loc='upper left')
    st.pyplot()


    plt.title(stock + " MACD and Signal line")
    plt.figure(figsize=(16, 4))
    plt.plot(data['MACD'], label='MACD', alpha=0.5)
    plt.plot(data['Signal Line'], label='Signal Line', alpha=0.5)
    plt.legend(loc='upper left')
    st.pyplot()

# NEURAL NET:___________________________
elif option == "Future price prediction":

    st.title("Neural networks and predicting future prices using Long-Short Term memory networks.")
    st.write("What are LSTM layers? \n An LSTM layer above provides a sequence output rather than a single value output to the LSTM layer below. Specifically, one output per input time step, rather than one output time step for all input time steps.")
    st.write("Why use more LSTM layers? \n Given that LSTMs operate on sequence data, it means that the addition of layers adds levels of abstraction of input observations over time. In effect, chunking observations over time or representing the problem at different time scales.")
    st.write("Read further: https://machinelearningmastery.com/stacked-long-short-term-memory-networks")
    st.write("Learn more about how choosing different data or splitting the training and testing data differently can effect your neural network results: \n https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6")
    st.write("Please wait while the model is being trained. (Too much training data can take a while to train. With reasonable settings, it should take less than a minute!)")
    
    st.sidebar.header('Choose Data to test:')
    start_date, end_date, stock = st.sidebar.text_input(
        "Start Date", "2018-01-01"), st.sidebar.text_input("End Date", "2020-01-01"), st.sidebar.text_input("Stock", "TSLA")
    df = get_data(stock, start_date, end_date)
    data = pd.DataFrame()

    trainingperc = int(st.sidebar.slider('How much of the data do you want to train on? (The rest will be used for evaluation.) (%)', 1, 100, 80, 1))
    trainingperc = trainingperc/100
    trainingtestperc = int(st.sidebar.slider('Of the training data, how much do you want to save use as input? (The rest will be used as expected output (labels)) (%)', 1, 100, 50, 1))
    trainingtestperc = trainingtestperc/100

    nLSTMlayers = int(st.sidebar.slider('How many LSTM layers do you want the model to have?', 1, 10, 2, 1))
    nofLSTMneurons = int(st.sidebar.slider('How many neurons should each LSTM layer have?', 1, 100, 50, 1))
    nDenselayers = int(st.sidebar.slider('How many Dense layers do you want the model to have?', 1, 10, 1, 1))
    nofDenseneurons = int(st.sidebar.slider('How many neurons should each dense layer have?', 1, 100, 25, 1))
    # st.stop()


    st.write("Network architecture:")
    # Build Visual  model
    st.write("__Layer (type)                 (Neurons)__")
    for i in range(0,nLSTMlayers):
      string = str("    lstm_"+str(i)+" (LSTM)                 ("+str(nofLSTMneurons)+")")
      st.write(string)
    for i in range(0,nDenselayers):
      string = str("    dense"+str(i)+" (Dense)                 ("+str(nofDenseneurons)+")")
      st.write(string)
    string = str("    Final_dense (Dense)                 (1)")
    st.write(string)

    # Create a new dataframe with only the 'Close' column
    data = df.filter(['Close'])
    # Converting the dataframe to a numpy array
    dataset = data.values
    # Get /Compute the number of rows to train the model on
    training_data_len = math.ceil(len(dataset) * trainingperc)
    # Scale the all of the data to be values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    # Create the scaled training data set
    train_data = scaled_data[0:training_data_len, :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []
    trainingtestperc = math.ceil(training_data_len*trainingtestperc)
    for i in range(trainingtestperc, len(train_data)):
        x_train.append(train_data[i-trainingtestperc:i, 0])
        y_train.append(train_data[i, 0])

    # Convert x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data into the shape accepted by the LSTM
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM network model
    model = Sequential()
    for i in range(0,nLSTMlayers):
      model.add(LSTM(units=nofLSTMneurons, return_sequences=True,input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=nofLSTMneurons, return_sequences=False))
    for i in range(0,nDenselayers):
      model.add(Dense(units=nofDenseneurons))
    model.add(Dense(units=1))

    # model = Sequential()
    # model.add(LSTM(units=50, return_sequences=True,
    #                input_shape=(x_train.shape[1], 1)))
    # model.add(LSTM(units=50, return_sequences=False))
    # model.add(Dense(units=25))
    # model.add(Dense(units=1))
    # print(model.summary())

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Test data set
    test_data = scaled_data[training_data_len - 60:, :]
    # Create the x_test and y_test data sets
    x_test = []
    # Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    # Convert x_test to a numpy array
    x_test = np.array(x_test)
    # Reshape the data into the shape accepted by the LSTM
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Getting the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)  # Undo scaling

    # Plot/Create the data for the graph
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    # Visualize the data
    st.header(stock + " Prediction backtest")
    rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
    st.write("Evaluation indicators: \n RMSE: " + str(rmse)+" (lower is better)")
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    st.pyplot()
