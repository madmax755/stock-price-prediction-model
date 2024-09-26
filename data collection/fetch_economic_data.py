import pandas as pd
import requests
from datetime import datetime
import os
from dotenv import load_dotenv
from typing import Dict, Optional

load_dotenv()

def fetch_economic_data(start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch economic indicator data from FRED (Federal Reserve Economic Data) API.

    This function retrieves data for GDP, Unemployment Rate, and Federal Funds Rate
    for the specified date range. It uses the FRED API and requires an API key
    to be set in the environment variables.

    Parameters:
    -----------
    start_date : str
        The start date for the data range in 'YYYY-MM-DD' format.
    end_date : str, optional
        The end date for the data range in 'YYYY-MM-DD' format.
        If not provided, the current date is used.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the economic indicator data. The index is the date,
        and columns include 'GDP', 'Unemployment_Rate', and 'Federal_Funds_Rate'.

    Notes:
    ------
    - Requires a FRED API key to be set as an environment variable 'FRED_API_KEY'.
    - GDP data is typically available quarterly, while other indicators may be monthly.
    - Missing values are filled with NaN.

    Example:
    --------
    >>> df = fetch_economic_data('2020-01-01', '2023-06-30')
    >>> print(df.head())
    """
    
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        raise ValueError("FRED API key not found in environment variables.")

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    indicators: Dict[str, str] = {
        "GDP": "GDP",
        "Unemployment_Rate": "UNRATE",
        "Federal_Funds_Rate": "FEDFUNDS"
    }

    def fetch_indicator(indicator: str, start: str, end: str) -> pd.Series:
        base_url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": indicator,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end
        }
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        data = response.json()
        df = pd.DataFrame(data['observations'])
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df.set_index('date')['value']

    economic_data = pd.DataFrame()
    for name, code in indicators.items():
        series = fetch_indicator(code, start_date, end_date)
        economic_data[name] = series
        print(f"Data for {name} fetched successfully.")

    economic_data.index.name = 'date'
    return economic_data.sort_index()

# Example usage
if __name__ == "__main__":
    start_date = "2018-01-01"
    end_date = "2023-06-30"
    df = fetch_economic_data(start_date, end_date)
    print(df.head())
    print(f"Data shape: {df.shape}")