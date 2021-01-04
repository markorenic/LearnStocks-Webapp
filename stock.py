from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.dates as mdates
from io import BytesIO
import base64
import gc
import time

api_key = 'WYJWWKPL9VO4I64B'

# ts = TimeSeries(key=api_key, output_format='pandas')
# data, metadata = ts.get_intraday(symbol = 'MSFT', interval = '1min', outputsize = 'full')
# percent_change = close_data.pct_change()
# print(percent_change)

def incoming_data(stock):

    # config.py stores the Alpha Vantage API key (https://www.alphavantage.co/support/#api-key)
    ts = TimeSeries(key = api_key)

    # compact is prior 100 days of data
    data_recent = ts.get_daily_adjusted(stock, outputsize = 'compact')

    # full is up to 20 years of data (if available)
    data_old = ts.get_daily_adjusted(stock, outputsize = 'full')

    date_recent = []
    close_recent = []

    date_old = []
    close_old = []

    now = datetime.now()

    # specifies how many historical days to include on the recent chart
    lookback = 200

    # iterates through recent data and appends date & close lists with data and close data for lookback period
    for keys in data_recent[0].keys():
        if (datetime.toordinal(pd.to_datetime(keys)) >= (now.toordinal() - lookback)):
            date_recent.append(pd.to_datetime(keys))
            close_recent.append(float(data_recent[0][keys]['5. adjusted close']))

    # iterates through old data and appends date & close lists with data and close data from 2008
    for keys_old in data_old[0].keys():
        if int(keys_old[0:4]) > 2008:
            date_old.append(pd.to_datetime(keys_old))
            close_old.append(float(data_old[0][keys_old]['5. adjusted close']))

    df_recent = pd.DataFrame({'date': date_recent, 'close': close_recent})

    df_old = pd.DataFrame({'date': date_old, 'close': close_old})

    df_recent.sort_values('date', ascending = False, inplace = True)
    df_old.sort_values('date', ascending = False, inplace = True)

    df_recent['date'] = pd.to_datetime(df_recent['date'])
    df_old['date'] = pd.to_datetime(df_old['date'])

    df_recent.set_index(['date'], inplace = True)
    df_old.set_index(['date'], inplace = True)

    return df_recent, df_old

def chartstring_stocks(stock):

    style.use('seaborn-darkgrid')

    df, df_old = incoming_data(stock)

    fig1, ax1 = plt.subplots( figsize = (18, 6))

    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # vertical line indicating date of purchase
    purchase_line = ax1.axvline(pd.to_datetime('2018-02-26'), ls='--', color='#FF9700', label='Purchase date')
    ax1.legend(handles=[purchase_line])

    fig1.autofmt_xdate()

    ax1.plot(df.index, df['close'], 'g')
    ax2.plot(df_old.index, df_old['close'], 'g')

    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    ax1.set_title('200 days ago --> Present')
    ax2.set_title('10 years ago --> Present')

    ax1.set_xlabel('Date')
    ax2.set_xlabel('Date')

    ax1.set_ylabel('Value (per share)')
    ax2.set_ylabel('Value (per share)')

    figfile = BytesIO()
    plt.savefig(figfile, format='png', dpi=(100))

    figfile.seek(0)

    figdata_png = base64.b64encode(figfile.getvalue()).decode('ascii')

    result = str(figdata_png)  # [2:-1]

    plt.show()

    plt.close()

    return result

#ascii_chart contains a string of the chart produced by matplotlib in ASCII
ascii_chart = chartstring_stocks('aapl')