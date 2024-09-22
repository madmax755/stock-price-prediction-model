import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    data.index = data.index.tz_localize(None)
    return data

def main():
    # Define parameters
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']  
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)  # 5 years of data

    # Fetch and save data for each ticker
    for ticker in tickers:
        data = fetch_stock_data(ticker, start_date, end_date)
        data.to_csv(f'data/{ticker}_data.csv')
        print(f"Data for {ticker} saved successfully.")

if __name__ == "__main__":
    main()