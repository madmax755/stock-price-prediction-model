import pandas as pd
import requests
from typing import Dict, Any

def fetch_treasury_yield_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches U.S. Treasury yield data for a specified date range from the U.S. Treasury API.

    This function retrieves daily treasury yield curve rates for various securities
    over the specified date range. It uses the U.S. Treasury's public API to fetch the data
    and returns it as a pandas DataFrame with dates as the index and securities as columns.

    Parameters:
    -----------
    start_date : str
        The start date for the data range in 'YYYY-MM-DD' format.
    end_date : str
        The end date for the data range in 'YYYY-MM-DD' format.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the treasury yield data. The index is the date,
        and each column represents a different security (e.g., '1 Month', '3 Month', etc.).
        The values are the average interest rates for each security on each date.

    Notes:
    ------
    - The function uses the 'avg_interest_rates' endpoint of the U.S. Treasury API.
    - The data is sorted in descending order by date.
    - If the API request fails or returns no data, an empty DataFrame is returned.

    Example:
    --------
    >>> yields = fetch_treasury_yield_data('2023-01-01', '2023-06-30')
    >>> print(yields.head())
    """
    base_url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/avg_interest_rates"
    
    params: Dict[str, Any] = {
        "fields": "record_date,security_desc,avg_interest_rate_amt",
        "filter": f"record_date:gte:{start_date},record_date:lte:{end_date}",
        "sort": "-record_date",
        "format": "json"
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data['data'])
        df['record_date'] = pd.to_datetime(df['record_date'])
        df['avg_interest_rate_amt'] = df['avg_interest_rate_amt'].astype(float)
        
        # Pivot the data to have securities as columns
        df_pivoted = df.pivot(index='record_date', columns='security_desc', values='avg_interest_rate_amt')
        
        return df_pivoted.sort_index()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Treasury yield data: {str(e)}")
        return pd.DataFrame()

if __name__ == "__main__":
    start_date = "2023-01-01"
    end_date = "2023-06-01"
    treasury_data = fetch_treasury_yield_data(start_date, end_date)
    print(treasury_data.head())
    print(f"Data shape: {treasury_data.shape}")