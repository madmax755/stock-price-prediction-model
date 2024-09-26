import yfinance as yf
import requests
import pandas as pd
from typing import Dict, Any
import time
from dotenv import load_dotenv
import os
from ratelimit import limits, sleep_and_retry

load_dotenv()  # Load environment variables from .env file

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
IEX_CLOUD_API_KEY = os.getenv('IEX_CLOUD_API_KEY')


@sleep_and_retry
@limits(calls=5, period=60)  # Limit to 5 calls per minute
def fetch_yahoo_finance_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical stock data for a given symbol from Yahoo Finance.

    This function uses the yfinance library to retrieve daily stock data
    including open, high, low, close prices, and volume for the specified date range.

    Parameters:
    -----------
    symbol : str
        The stock symbol to fetch data for (e.g., 'AAPL' for Apple Inc.).
    start_date : str
        The start date for the data range in 'YYYY-MM-DD' format.
    end_date : str
        The end date for the data range in 'YYYY-MM-DD' format.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the stock data. The index is the date,
        and columns include 'Open', 'High', 'Low', 'Close', 'Volume', and other available metrics.

    Example:
    --------
    >>> data = fetch_yahoo_finance_data('AAPL', '2023-01-01', '2023-06-30')
    >>> print(data.head())
    """
    
    try:
        data = yf.Ticker(symbol).history(start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"Error fetching data from Yahoo Finance for {symbol}: {str(e)}")
        return pd.DataFrame()

@sleep_and_retry
@limits(calls=5, period=60)  # Limit to 5 calls per minute
def fetch_alpha_vantage_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical stock data for a given symbol from Alpha Vantage.

    This function uses the Alpha Vantage API to retrieve daily stock data
    including open, high, low, close prices, and volume. It then filters
    the data to the specified date range.

    Parameters:
    -----------
    symbol : str
        The stock symbol to fetch data for (e.g., 'AAPL' for Apple Inc.).
    start_date : str
        The start date for the data range in 'YYYY-MM-DD' format.
    end_date : str
        The end date for the data range in 'YYYY-MM-DD' format.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the stock data. The index is the date,
        and columns include 'Open', 'High', 'Low', 'Close', and 'Volume'.

    Notes:
    ------
    - Requires an Alpha Vantage API key set as an environment variable.
    - The function fetches the full available dataset and then filters to the specified date range.

    Example:
    --------
    >>> data = fetch_alpha_vantage_data('AAPL', '2023-01-01', '2023-06-30')
    >>> print(data.head())
    """

    base_url = "https://www.alphavantage.co/query"
    function = "TIME_SERIES_DAILY"
    
    params: Dict[str, Any] = {
        "function": function,
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY,
        "outputsize": "full"
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        # gets the actual data from the dictionary containing both metadata and the requested time series data
        time_series = data.get("Time Series (Daily)", {})
        

        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)

        df = df.sort_index()

        # Filter the dataframe to the desired date range (using a mask to be more flexible if threshold values dont exist)
        df = df.loc[(df.index >= start_date) & (df.index <= end_date)]

        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Alpha Vantage for {symbol}: {str(e)}")
        return pd.DataFrame()

def fetch_data_from_all_sources(symbol: str, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Fetches historical stock data for a given symbol from multiple sources.

    This function retrieves stock data from Yahoo Finance andAlpha Vantage
    for the specified symbol and date range. It returns a dictionary containing
    separate DataFrames for each data source.

    Parameters:
    -----------
    symbol : str
        The stock symbol to fetch data for (e.g., 'AAPL' for Apple Inc.).
    start_date : str
        The start date for the data range in 'YYYY-MM-DD' format.
    end_date : str
        The end date for the data range in 'YYYY-MM-DD' format.

    Returns:
    --------
    Dict[str, pd.DataFrame]
        A dictionary containing DataFrames with the stock data from each source.
        The keys are 'yahoo' and 'alpha_vantage' and the values are
        the corresponding DataFrames. Each DataFrame has a DatetimeIndex and
        includes columns for Open, High, Low, Close, and Volume (column names
        may vary slightly between sources).

    Raises:
    -------
    Exception
        If there's an error in fetching or processing the data from any source.
        Note that errors from individual sources are caught and logged, and will
        not prevent data from other sources from being returned.

    Notes:
    ------
    - Requires API keys for Alpha Vantage to be set as environment variables.
    - If a particular source fails to retrieve data, its corresponding DataFrame
      in the returned dictionary will be empty.
    - The function attempts to fetch data from all sources even if one or more fail.

    Example:
    --------
    >>> all_data = fetch_data_from_all_sources('AAPL', '2023-01-01', '2023-06-30')
    >>> for source, data in all_data.items():
    ...     print(f"\nData from {source}:")
    ...     print(data.head())
    ...     print(f"Shape: {data.shape}")
    """

    yahoo_data = fetch_yahoo_finance_data(symbol, start_date, end_date)
    alpha_vantage_data = fetch_alpha_vantage_data(symbol, start_date, end_date)
    
    return {
        "yahoo": yahoo_data,
        "alpha_vantage": alpha_vantage_data,
    }

if __name__ == "__main__":
    symbol = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-06-01"
    
    all_data = fetch_data_from_all_sources(symbol, start_date, end_date)
    
    for source, data in all_data.items():
        print(f"\nData from {source}:")
        print(data.head())
        print(f"Shape: {data.shape}")