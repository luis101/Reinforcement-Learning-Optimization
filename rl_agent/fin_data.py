
"""
Get financial data for S&P 500 firms, including historical stock prices, dividends, and volumes. 
The data is filtered to include only stocks that were part of the S&P 500 index at the respective dates. 
The resulting dataset is stored in CSV format for further analysis and use in reinforcement learning models.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import gc

#import mplfinance as mpf
import yfinance as yf


# %% Import S&P 500 firms

def get_sp500(url = "https://raw.githubusercontent.com/fja05680/sp500/refs/heads/master/S%26P%20500%20Historical%20Components%20%26%20Changes(01-17-2026).csv"):

    sp500 = pd.read_csv(url, delimiter=",")
    print(sp500.dtypes)

    ticker = set([s for s in sp500.iloc[1,1].split(",")])

    for i in range(2,len(sp500)):
        t = set([s for s in sp500.iloc[i,1].split(",")])
        for ele in t:
            ticker.add(ele)

    print(len(ticker))
    #print(ticker)

    sp500[['year','month','day']] = sp500['date'].str.split('-', expand=True)

    return ticker, sp500


# %% Download data

def download_fin_data(ticker, start_date = "2005-01-01", month_end=True, end_date=None, sp500=None):
    
    if end_date is None:
        if month_end:
            # Get current date
            end_date = datetime.today().strftime('%Y-%m-%d')
        else:    
            # Get last day of previous month
            end_date = (pd.Timestamp.today() - pd.tseries.offsets.BMonthEnd(1)).strftime('%Y-%m-%d')

    stocks_daily = pd.DataFrame()
    stock_prices = pd.DataFrame()
    stocks = pd.DataFrame()

    # Define the stock symbol and loop over symbols

    for stock_symbol in ticker:
        
        # Download historical data via yf.download (adjusted close price, dividends, volume)
        
        print("Ticker: " + stock_symbol)

        try:
            stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
            if stock_data.index.tz is not None:
                stock_data.index = stock_data.index.tz_localize(None)
            stock_data = stock_data.stack(1)
            stock_data = stock_data.reset_index(level=1)
            
            stock_data['month_id'] = stock_data.index.strftime('%Y-%m')
            stock_data['numst'] = stock_data.groupby(['month_id'])['Ticker'].transform('count')
            stock_data = stock_data[(stock_data['numst']>=17)]

            #stocklist.append(sdf)  
            #st_data = stock_data.groupby(['month_id']).last().reset_index()
            stocks_daily = pd.concat([stocks_daily, stock_data], axis=1)
        except Exception as e:
            print(f"Error downloading data for {stock_symbol}: {e}")

        # Load historical data via yf.Ticker.history (adjusted close price, dividends, volume)

        stock = yf.Ticker(stock_symbol)
        try:
            data = stock.history(period="max")
        except:
            continue
        
        if len(data) == 0:
            continue

        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        data['Ticker'] = stock_symbol

        data['month_id'] = data.index.strftime('%Y-%m')
        data[['Vol', 'Div']] = data.groupby(['month_id'])[['Volume', 'Dividends']].transform('sum')
        data['numst'] = data.groupby(['month_id'])['Ticker'].transform('count')
        data['Price'] = data['Close'] + data['Div']

        # Keep only the last day of each month and calculate monthly returns, 
        # but only for months with at least 17 trading days (to ensure liquidity)

        sdf = data.groupby(['month_id']).last().reset_index()
        sdf["return"] = ((sdf["Close"]+sdf['Div']) - sdf["Close"].shift(1)) / sdf["Close"].shift(1)
        sdf = sdf[(sdf['numst']>=17)]

        sdf = sdf[['month_id', 'Ticker', 'Close', 'Volume', 'Div', 'return']]

        stocks = pd.concat([stocks, sdf], axis=0)

        # Filter to include only stocks that were part of the S&P 500 index at the respective dates 

        if sp500 is not None:
            data = _current_index(data, sp500)

        data.rename(columns={'Price':stock_symbol}, inplace=True)

        prices = data[[stock_symbol]].copy()
        stock_prices = pd.concat([stock_prices, prices], axis=1)

        #del sdf
        #gc.collect()

    return stocks, stocks_daily, stock_prices

# Only if in S&P500 at the respective date

def _current_index(stocks, sp500):

    sp500['month_id'] = pd.to_datetime(sp500['date']).dt.strftime('%Y-%m')
    sp = sp500.groupby(['month_id']).last().reset_index()

    # Example: Replace 'Ticker' and 'Value' with actual column names, e.g., 'date' and 'Symbol'
    # ticker_ts = sp500.pivot(index='date', columns='Symbol', values='SomeValueColumn')
    sp_long = sp[['month_id', 'tickers']].copy()
    sp_long = sp_long.dropna(subset=['tickers'])
    sp_long['tickers'] = sp_long['tickers'].str.split(',')
    sp_long = sp_long.explode('tickers')
    #sp_long['tickers'] = sp_long['tickers'].str.strip()
    sp_long = sp_long.reset_index(drop=True)
    sp_long.rename(columns={'tickers':'Ticker'}, inplace=True)

    # stocks_sp = stocks.merge(sp_long, on=['month_id', 'Ticker'], how='inner')
    stocks_sp = stocks.reset_index().merge(sp_long, on=['month_id', 'Ticker'], how='inner').set_index('Date')

    return stocks_sp

